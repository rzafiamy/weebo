import json
import re
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import onnxruntime
import phonemizer
import sounddevice as sd
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import espeakng_loader
from ollama import chat
from faster_whisper import WhisperModel
import signal
from threading import Event
import torch
import logging

# Configuration du logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Weebo:
    def __init__(self):
        logging.info("Initialisation de Weebo...")
        # audio settings
        self.SAMPLE_RATE = 24000
        self.WHISPER_SAMPLE_RATE = 16000
        self.SILENCE_THRESHOLD = 0.02   # volume level that counts as silence
        self.SILENCE_DURATION = 1.5    # seconds of silence before cutting recording

        # text-to-speech settings
        self.MAX_PHONEME_LENGTH = 510
        self.CHUNK_SIZE = 300         # size of text chunks for processing
        self.SPEED = 1.2
        self.VOICE = "am_michael"

        # processing things
        self.MAX_THREADS = 1

        # ollama settings
        self.messages = []
        self.SYSTEM_PROMPT = "Give a conversational response to the following statement or question in 1-2 sentences. The response should be natural and engaging, and the length depends on what you have to say."

        # init components
        self._init_espeak()
        self._init_models()
        self.executor = ThreadPoolExecutor(max_workers=self.MAX_THREADS)

        # interrupt handling
        self.shutdown_event = Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        logging.info("Weebo initialisé avec succès!")

    def _signal_handler(self, signum, frame):
        logging.warning("Interruption détectée, arrêt en cours...")
        self.shutdown_event.set()

    def _init_espeak(self):
        logging.debug("Initialisation de eSpeak...")
        # setup espeak for phoneme generation
        espeak_data_path = espeakng_loader.get_data_path()
        espeak_lib_path = espeakng_loader.get_library_path()
        EspeakWrapper.set_data_path(espeak_data_path)
        EspeakWrapper.set_library(espeak_lib_path)

        # vocab for phoneme tokenization
        self.vocab = self._create_vocab()

    def _init_models(self):
        logging.debug("Chargement des modèles...")
        # init text-to-speech model
        self.tts_session = onnxruntime.InferenceSession(
            "kokoro-v0_19.onnx",
            providers=["CPUExecutionProvider"]
        )

        # load voice profiles
        with open("voices.json") as f:
            self.voices = json.load(f)

        # Load Whisper for Speech-to-Text
        model_size = "medium"  # You can use "medium", "large-v3", etc.
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.whisper_model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")



    def _create_vocab(self) -> Dict[str, int]:
        # create mapping of characters/phonemes to integer tokens
        chars = ['$'] + list(';:,.!?¡¿—…"«»"" ') + \
            list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") + \
            list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
        return {c: i for i, c in enumerate(chars)}

    def phonemize(self, text: str) -> str:
        # clean text and convert to phonemes
        text = re.sub(r"[^\S \n]", " ", text)
        text = re.sub(r"  +", " ", text).strip()
        phonemes = phonemizer.phonemize(
            text,
            "en-us",
            preserve_punctuation=True,
            with_stress=True
        )
        return "".join(p for p in phonemes.replace("r", "ɹ") if p in self.vocab).strip()

    def generate_audio(self, phonemes: str, voice: str, speed: float) -> np.ndarray:
        logging.debug(f"Génération audio pour phonèmes: {phonemes}")
        # convert phonemes to audio using TTS model
        tokens = [self.vocab[p] for p in phonemes if p in self.vocab]
        if not tokens:
            return np.array([], dtype=np.float32)

        tokens = tokens[:self.MAX_PHONEME_LENGTH]
        style = np.array(self.voices[voice], dtype=np.float32)[len(tokens)]

        audio = self.tts_session.run(
            None,
            {
                'tokens': [[0, *tokens, 0]],
                'style': style,
                'speed': np.array([speed], dtype=np.float32)
            }
        )[0]

        return audio

    def record_and_transcribe(self):
        logging.info("Enregistrement en cours...")
        audio_buffer = []
        silence_frames = 0

        def callback(indata, frames, time_info, status):
            if self.shutdown_event.is_set():
                raise sd.CallbackStop()
            
            audio = indata.flatten()
            level = np.abs(audio).mean()
            audio_buffer.extend(audio.tolist())

            if level < self.SILENCE_THRESHOLD:
                nonlocal silence_frames
                silence_frames += len(audio)
            else:
                silence_frames = 0

            if silence_frames > self.SILENCE_DURATION * self.SAMPLE_RATE:
                audio_segment = np.array(audio_buffer, dtype=np.float32)
                if len(audio_segment) > self.SAMPLE_RATE:
                    segments, _ = self.whisper_model.transcribe(audio_segment, beam_size=5)
                    segments = list(segments)  # Ensures the transcription runs before processing
                    text = " ".join(segment.text for segment in segments)
                    if text.strip():
                        logging.info("-------------------------------------------")
                        logging.info(f"USER: {text}")
                        logging.info("-------------------------------------------")
                        self.create_and_play_response(text)
                audio_buffer.clear()
                silence_frames = 0

        with sd.InputStream(callback=callback, channels=1, samplerate=self.SAMPLE_RATE, dtype=np.float32):
            print("Recording... Press Ctrl+C to stop")
            while not self.shutdown_event.is_set():
                sd.sleep(100)

    def create_and_play_response(self, prompt: str):
        logging.info(f"Génération de réponse pour: {prompt}")
        if self.shutdown_event.is_set() or prompt.strip() == "":
            return

        # stream response from llm
        stream = chat(
            model='gemma3:1B',
            messages=[{
                'role': 'system',
                'content': self.SYSTEM_PROMPT
            }, {
                'role': 'user',
                'content': prompt
            }] + self.messages,
            stream=True,
        )

        # state for processing response
        futures = []
        buffer = ""
        bot = ""
        curr_str = ""

        try:
            # process response stream
            for chunk in stream:
                if self.shutdown_event.is_set():
                    break

                # print(chunk)
                text = chunk['message']['content']

                bot += text

                if len(text) == 0:
                    self.messages.append({
                        'role': 'assistant',
                        'content': curr_str
                    })
                    curr_str = ""
                    #print(self.messages)
                    continue

                buffer += text
                curr_str += text

                # find end of sentence to chunk at
                last_punctuation = max(
                    buffer.rfind('. '),
                    buffer.rfind('? '),
                    buffer.rfind('! ')
                )

                if last_punctuation == -1:
                    continue

                # handle long chunks
                while last_punctuation != -1 and last_punctuation >= self.CHUNK_SIZE:
                    last_punctuation = max(
                        buffer.rfind(', ', 0, last_punctuation),
                        buffer.rfind('; ', 0, last_punctuation),
                        buffer.rfind('— ', 0, last_punctuation)
                    )

                if last_punctuation == -1:
                    last_punctuation = buffer.find(' ', 0, self.CHUNK_SIZE)

                # process chunk
                # convert chunk to audio
                chunk_text = buffer[:last_punctuation + 1]
                ph = self.phonemize(chunk_text)
                futures.append(
                    self.executor.submit(
                        self.generate_audio,
                        ph, self.VOICE, self.SPEED
                    )
                )
                buffer = buffer[last_punctuation + 1:]

            # process final chunk if any
            if buffer and not self.shutdown_event.is_set():
                ph = self.phonemize(buffer)
                futures.append(
                    self.executor.submit(
                        self.generate_audio,
                        ph, self.VOICE, self.SPEED
                    )
                )

            # play generated audio
            if not self.shutdown_event.is_set():
                logging.info("-----------------------------------")
                logging.info(f"Bot: {bot}")
                logging.info("-----------------------------------")
                with sd.OutputStream(
                    samplerate=self.SAMPLE_RATE,
                    channels=1,
                    dtype=np.float32
                ) as out_stream:
                    for fut in futures:
                        if self.shutdown_event.is_set():
                            break
                        audio_data = fut.result()
                        if len(audio_data) == 0:
                            continue
                        out_stream.write(audio_data.reshape(-1, 1))
        except Exception as e:
            if not self.shutdown_event.is_set():
                raise e


def main():
    weebo = Weebo()
    weebo.record_and_transcribe()


if __name__ == "__main__":
    main()
