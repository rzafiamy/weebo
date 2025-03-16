import logging
import signal
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from core.tts import TextToSpeech
from core.speech import SpeechProcessor
from core.chatbot import ChatBot
import sounddevice as sd
import numpy as np
import config

# Logging Configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class Weebo:
    def __init__(self):
        logging.info("Initializing Weebo...")
        self.tts = TextToSpeech()
        self.speech_processor = SpeechProcessor()
        self.chatbot = ChatBot()
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_THREADS)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logging.warning("Interruption detected, shutting down...")
        self.speech_processor.shutdown_event.set()

    def handle_user_input(self, text: str):
        response = self.chatbot.generate_response(text)
        logging.info("--------------")
        logging.info(f"BOT: {response}")
        logging.info("--------------")
        phonemes = self.tts.phonemize(response)
        audio = self.tts.generate_audio(phonemes, config.VOICE, config.SPEED)
        self.play_audio(audio)

    def play_audio(self, audio_data):
        with sd.OutputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype=np.float32) as out_stream:
            out_stream.write(audio_data.reshape(-1, 1))

    def start(self):
        self.speech_processor.record_and_transcribe(self.handle_user_input)


if __name__ == "__main__":
    weebo = Weebo()
    weebo.start()
