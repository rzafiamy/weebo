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

# Logging Configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TextToSpeech:
    def __init__(self):
        logging.info("Initializing TextToSpeech...")
        self.tts_session = onnxruntime.InferenceSession("kokoro-v0_19.onnx", providers=["CPUExecutionProvider"])
        self.vocab = self._create_vocab()
        self._init_espeak()
        with open("voices.json") as f:
            self.voices = json.load(f)

    def _init_espeak(self):
        espeak_data_path = espeakng_loader.get_data_path()
        espeak_lib_path = espeakng_loader.get_library_path()
        EspeakWrapper.set_data_path(espeak_data_path)
        EspeakWrapper.set_library(espeak_lib_path)

    def _create_vocab(self) -> Dict[str, int]:
        # create mapping of characters/phonemes to integer tokens
        chars = ['$'] + list(';:,.!?¡¿—…"«»"" ') + \
            list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") + \
            list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
        return {c: i for i, c in enumerate(chars)}

    def phonemize(self, text: str) -> str:
        text = re.sub(r"[^\S \n]", " ", text).strip()
        phonemes = phonemizer.phonemize(text, "en-us", preserve_punctuation=True, with_stress=True)
        return "".join(p for p in phonemes if p in self.vocab).strip()

    def generate_audio(self, phonemes: str, voice: str, speed: float) -> np.ndarray:
        # convert phonemes to audio using TTS model
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
