import signal
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from core.tts import TextToSpeech
from core.speech import SpeechProcessor
from core.chatbot import ChatBot
from core.rag import RAG
import sounddevice as sd
import numpy as np
import config
import logging
import time
import os

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Weebo:
    def __init__(self, context):
        logging.info("Initializing Weebo...")
        self.tts = TextToSpeech()
        self.speech_processor = SpeechProcessor()
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_THREADS)
        self.file  = context
        self.rag = RAG()  # Initialize the RAG system

        if self.file and os.path.exists(self.file):
            logging.info(f"File {self.file} exists. Ingesting...")
            self.rag.ingest_pdfs([self.file])  # Ingest the PDF file
        
        self.chatbot = ChatBot(self.rag)

        self.audio_playing_event = Event()  # Blocks recording when audio is playing
        self.processing_lock = Lock()  # Ensures chatbot processing is sequential

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signal to shut down the Weebo instance."""
        logging.warning("Interruption detected, shutting down...")
        self.speech_processor.shutdown_event.set()

    def handle_user_input(self, text: str):
        """Process user input -> chatbot -> play response."""
        with self.processing_lock:  # Ensures strict order of execution
            logging.info("--------------")
            logging.info(f"USER: {text}")
            logging.info("--------------")

            response = self.chatbot.generate_response(text)

            logging.info("--------------")
            logging.info(f"BOT: {response}")
            logging.info("--------------")

            phonemes = self.tts.phonemize(response)
            audio = self.tts.generate_audio(phonemes, config.VOICE, config.SPEED)

            self.play_audio(audio)  # Block recording while playing

    def play_audio(self, audio_data):
        """Stops recording, plays audio, and restarts recording."""
        self.audio_playing_event.set()  # 🔴 Stop recording
        logging.info("🔊 Playing audio... Recording is stopped.")

        with sd.OutputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype=np.float32) as out_stream:
            out_stream.write(audio_data.reshape(-1, 1))

        logging.info("✅ Audio playback finished. Restarting recording.")

        time.sleep(0.2)  # Small delay before restarting
        self.audio_playing_event.clear()  # 🟢 Allow recording to start again


    def start(self):
        """Start recording and sequentially process speech input."""
        self.speech_processor.record_and_transcribe(self.handle_user_input, self.audio_playing_event)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG System for PDF Ingestion and Querying")
    parser.add_argument("--file")
    
    args = parser.parse_args()
    
    weebo = Weebo(args.file)
    weebo.start()
