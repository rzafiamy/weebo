import torch
from threading import Event
from faster_whisper import WhisperModel
import config
import logging
import sounddevice as sd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Capture all levels of logs
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler("speech_processor.log")  # Logs to file
    ]
)

class SpeechProcessor:
    def __init__(self):
        """Initializes the SpeechProcessor with configuration parameters and loads the Whisper model."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing SpeechProcessor...")
        
        try:
            self.sample_rate = config.SAMPLE_RATE
            self.silence_threshold = config.SILENCE_THRESHOLD
            self.silence_duration = config.SILENCE_DURATION
            self.device = config.DEVICE
        except AttributeError as e:
            self.logger.error(f"Missing configuration value: {e}")
            raise

        self.shutdown_event = Event()
        self.whisper_model = self._load_whisper_model()

    def _load_whisper_model(self):
        """Loads the Whisper speech-to-text model with an appropriate device setting."""
        try:
            model_size = config.ASR_MODEL_SIZE
            compute_type = config.COMPUTE_TYPE
            self.logger.info(f"Loading Whisper model: {model_size} on {self.device} with compute type {compute_type}")
            return WhisperModel(model_size, device=self.device, compute_type=compute_type)
        except AttributeError as e:
            self.logger.error(f"Missing configuration value: {e}")
            raise

    def record_and_transcribe(self, callback):
        """Records audio from the microphone and transcribes speech when silence is detected."""
        self.logger.info("Starting recording...")
        audio_buffer = []  # Stores recorded audio samples
        silence_frames = 0  # Counter to track silence duration

        def audio_callback(indata, frames, time_info, status):
            """Processes audio input in real-time, detecting speech and silence."""
            if self.shutdown_event.is_set():
                raise sd.CallbackStop()

            if status:
                self.logger.warning(f"Input stream status: {status}")

            audio = indata.flatten()
            level = np.abs(audio).mean()
            audio_buffer.extend(audio.tolist())

            nonlocal silence_frames
            if level < self.silence_threshold:
                silence_frames += len(audio)  # Accumulate silence frames
            else:
                silence_frames = 0  # Reset silence counter when speech is detected

            # If silence exceeds the threshold, process the buffered audio
            if silence_frames > self.silence_duration * self.sample_rate:
                audio_segment = np.array(audio_buffer, dtype=np.float32)
                if len(audio_segment) > self.sample_rate:  # Ensure minimum length for transcription
                    try:
                        segments, _ = self.whisper_model.transcribe(audio_segment, beam_size=5)
                        text = " ".join(segment.text for segment in segments if segment.text.strip())
                        if text.strip():
                            self.logger.info("--------------------")
                            self.logger.info(f"USER: {text}")
                            self.logger.info("--------------------")
                            callback(text)  # Pass transcribed text to callback
                    except Exception as e:
                        self.logger.exception("Error occurred during transcription.")
                
                audio_buffer.clear()
                silence_frames = 0

        # Start recording with a non-blocking stream
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=self.sample_rate, dtype=np.float32):
            while not self.shutdown_event.is_set():
                sd.sleep(100)  # Sleep briefly to allow processing of the audio stream