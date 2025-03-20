import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Audio settings
ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "medium")  # Options: "medium", "large-v3", etc.
DEVICE = "cpu" if os.getenv("USE_CUDA", "True").lower() == "true" else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

WHISPER_SAMPLE_RATE = int(os.getenv("WHISPER_SAMPLE_RATE", 16000))
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", 0.04))
SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", 3))

# Text-to-Speech settings
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 24000))
MAX_PHONEME_LENGTH = int(os.getenv("MAX_PHONEME_LENGTH", 510))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))
SPEED = float(os.getenv("SPEED", 1.2))
VOICE = os.getenv("VOICE", "am_michael")
NO_VOICE_REPLY = os.getenv("NO_VOICE", "I'm sorry, I don't understand what you're asking.")

# Processing
MAX_THREADS = int(os.getenv("MAX_THREADS", 1))

# Ollama settings
MODEL=os.getenv("MODEL", "llama3.1:latest")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", 1))  # in seconds
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", 
    "Give a conversational response to the following statement or question in 1-2 sentences. The response should be natural and engaging, and the length depends on what you have to say.")


# Paths
TTS_MODEL_PATH = os.getenv("TTS_MODEL_PATH", "kokoro-v0_19.onnx")
VOICES_FILE = os.getenv("VOICES_FILE", "voices.json")


# Onnx
ONNX_DEVICE = os.getenv("ONNX_DEVICE", "CUDAExecutionProvider") # Options: "CPUExecutionProvider"


# EMBEDDINGS
CHROMA_DB=os.getenv("CHROMA_DB", "chroma_store")  
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")