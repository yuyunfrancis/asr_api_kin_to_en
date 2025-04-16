import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# API settings
API_TITLE = "Audio Transcription & Translation API"
API_DESCRIPTION = "API for transcribing and translating audio in Kinyarwanda and English"
API_VERSION = "1.0.0"

# Model settings
MODELS_CACHE_DIR = os.getenv("MODELS_CACHE_DIR", "./models_cache")
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN", "")

# File storage
TEMP_FILE_DIR = os.getenv("TEMP_FILE_DIR", "./temp_files")
TEMP_FILE_EXPIRY = int(os.getenv("TEMP_FILE_EXPIRY", "24"))  # hours

# Model-specific settings
WHISPER_MODEL_PATH = "openai/whisper-small"
KINYARWANDA_WHISPER_MODEL_PATH = "mbazaNLP/Whisper-Small-Kinyarwanda"
KINYARWANDA_NEMO_MODEL_PATH = "mbazaNLP/Kinyarwanda_nemo_stt_conformer_model"
KIN_TO_EN_TRANSLATION_MODEL = "RogerB/marian-finetuned-multidataset-kin-to-en"
EN_TO_KIN_TRANSLATION_MODEL = "mbazaNLP/Nllb_finetuned_education_en_kin"

# Language codes mapping
LANGUAGE_CODES = {
    "english": "en",
    "kinyarwanda": "sw"  # Whisper uses Swahili code for Kinyarwanda
}

# Check if torch is available (and GPU)
try:
    import torch
    torch_available = True
    gpu_available = torch.cuda.is_available()
except ImportError:
    torch_available = False
    gpu_available = False

# Resource management
USE_GPU = os.getenv("USE_GPU", "True").lower() == "true" and gpu_available

# Make sure output directories exist
os.makedirs(TEMP_FILE_DIR, exist_ok=True)
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_CACHE_DB = int(os.getenv("REDIS_CACHE_DB", "1"))

# Model caching
MAX_MODEL_CACHE_SIZE = int(os.getenv("MAX_MODEL_CACHE_SIZE", "3"))
