from phonemizer.backend.espeak.wrapper import EspeakWrapper
import espeakng_loader
import numpy as np
import onnxruntime
import phonemizer
import torch
import json
import re
import os
import config
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Capture all levels of logs
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler("speech_processor.log")  # Logs to file
    ]
)

class TextToSpeech:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing TextToSpeech...")
        
        try:
            self.tts_session = onnxruntime.InferenceSession(config.TTS_MODEL_PATH, providers=[config.ONNX_DEVICE])
            self.vocab = self._create_vocab()
            self._init_espeak()
            with open("voices.json") as f:
                self.voices = json.load(f)
        except Exception as e:
            self.logger.exception("Error initializing TextToSpeech.")
            raise

    def _init_espeak(self):
        try:
            espeak_data_path = espeakng_loader.get_data_path()
            espeak_lib_path = espeakng_loader.get_library_path()
            EspeakWrapper.set_data_path(espeak_data_path)
            EspeakWrapper.set_library(espeak_lib_path)
        except Exception as e:
            self.logger.exception("Error initializing eSpeak.")
            raise

    def _create_vocab(self) -> Dict[str, int]:
        chars = ['$'] + list(';:,.!?¡¿—…"«»"" ') + list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") + list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
        return {c: i for i, c in enumerate(chars)}

    def normalize(self, text: str) -> str:
        """
        Normalizes the input text by removing characters that are not in the allowed vocabulary.
        
        Parameters:
            text (str): The text to be normalized.
            
        Returns:
            str: The normalized text containing only allowed characters.
        """
        allowed_chars = set(self.vocab.keys())
        normalized_text = "".join(ch for ch in text if ch in allowed_chars)
        self.logger.debug(f"Normalized text: {normalized_text}")
        return normalized_text
        
    def phonemize(self, text: str) -> str:
        # Normalize the text first to filter out unwanted characters
        text = self.normalize(text)
        text = re.sub(r"[^\S \n]", " ", text).strip()
        phonemes = phonemizer.phonemize(text, "en-us", preserve_punctuation=True, with_stress=True)
        return "".join(p for p in phonemes if p in self.vocab).strip()


    def generate_audio(self, phonemes: str, voice: str, speed: float) -> np.ndarray:
        tokens = [self.vocab[p] for p in phonemes if p in self.vocab]
        if not tokens:
            return np.array([], dtype=np.float32)

        tokens = tokens[:config.MAX_PHONEME_LENGTH]
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
