import torch
from threading import Event
from faster_whisper import WhisperModel
import config
import logging
import sounddevice as sd
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



class SpeechProcessor:
    def __init__(self):
        logging.info("Initializing SpeechProcessor...")
        self.sample_rate = config.SAMPLE_RATE
        self.silence_threshold = config.SILENCE_THRESHOLD
        self.silence_duration = config.SILENCE_DURATION
        self.shutdown_event = Event()
        self.whisper_model = self._load_whisper_model()

    def _load_whisper_model(self):
        model_size = "medium"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")

    def record_and_transcribe(self, callback):
        logging.info("Starting recording...")
        audio_buffer = []
        silence_frames = 0

        def audio_callback(indata, frames, time_info, status):
            if self.shutdown_event.is_set():
                raise sd.CallbackStop()

            audio = indata.flatten()
            level = np.abs(audio).mean()
            audio_buffer.extend(audio.tolist())

            nonlocal silence_frames
            if level < self.silence_threshold:
                silence_frames += len(audio)
            else:
                silence_frames = 0

            if silence_frames > self.silence_duration * self.sample_rate:
                audio_segment = np.array(audio_buffer, dtype=np.float32)
                if len(audio_segment) > self.sample_rate:
                    segments, _ = self.whisper_model.transcribe(audio_segment, beam_size=5)
                    text = " ".join(segment.text for segment in segments if segment.text.strip())
                    if text.strip():
                        logging.info("--------------------")
                        logging.info(f"USER: {text}")
                        logging.info("--------------------")
                        callback(text)
                audio_buffer.clear()
                silence_frames = 0

        with sd.InputStream(callback=audio_callback, channels=1, samplerate=self.sample_rate, dtype=np.float32):
            while not self.shutdown_event.is_set():
                sd.sleep(100)