import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Audio settings
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 24000))
WHISPER_SAMPLE_RATE = int(os.getenv("WHISPER_SAMPLE_RATE", 16000))
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", 0.02))
SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", 1.5))

# Text-to-Speech settings
MAX_PHONEME_LENGTH = int(os.getenv("MAX_PHONEME_LENGTH", 510))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))
SPEED = float(os.getenv("SPEED", 1.2))
VOICE = os.getenv("VOICE", "am_michael")

# Processing
MAX_THREADS = int(os.getenv("MAX_THREADS", 1))

# Ollama settings
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", 
    "Give a conversational response to the following statement or question in 1-2 sentences. The response should be natural and engaging, and the length depends on what you have to say.")

# Model settings
MODEL_SIZE = os.getenv("MODEL_SIZE", "medium")  # Options: "medium", "large-v3", etc.
DEVICE = "cuda" if os.getenv("USE_CUDA", "False").lower() == "true" else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Paths
TTS_MODEL_PATH = os.getenv("TTS_MODEL_PATH", "kokoro-v0_19.onnx")
VOICES_FILE = os.getenv("VOICES_FILE", "voices.json")
