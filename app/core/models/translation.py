import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, Any, Tuple, Optional

from app import config

# Configure logging
logger = logging.getLogger("api.models")

# Cache for loaded models
TRANSLATION_MODEL_CACHE = {}

def load_translation_model(direction: str = "kin-to-en") -> Tuple[Any, Any]:
    """
    Load the translation model and tokenizer
    
    Args:
        direction: Translation direction, either "kin-to-en" or "en-to-kin"
        
    Returns:
        Tuple containing tokenizer and model for translation
    """
    cache_key = f"translation_{direction}"
    
    # Check if models are in cache
    if cache_key in TRANSLATION_MODEL_CACHE:
        logger.info(f"Using cached translation model for {direction}")
        return TRANSLATION_MODEL_CACHE[cache_key]
    
    if direction == "kin-to-en":
        model_name = config.KIN_TO_EN_TRANSLATION_MODEL
    else:  # en-to-kin
        model_name = config.EN_TO_KIN_TRANSLATION_MODEL
    
    logger.info(f"Loading translation model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=config.HUGGING_FACE_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=config.HUGGING_FACE_TOKEN)
    
    # Move to GPU if available
    if config.gpu_available:
        model = model.to("cuda")
        logger.info("Using GPU for translation")
        
    # Cache the models
    TRANSLATION_MODEL_CACHE[cache_key] = (tokenizer, model)
        
    return tokenizer, model

def get_language_direction(source_lang: str, target_lang: str) -> str:
    """
    Determine translation direction
    
    Args:
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        Translation direction string
    """
    if source_lang.lower() in ["kinyarwanda", "kin", "rw"] and target_lang.lower() in ["english", "en", "eng"]:
        return "kin-to-en"
    elif source_lang.lower() in ["english", "en", "eng"] and target_lang.lower() in ["kinyarwanda", "kin", "rw"]:
        return "en-to-kin"
    else:
        raise ValueError(f"Unsupported translation direction: {source_lang} to {target_lang}")

def get_language_codes(language: str, model_type: str) -> Tuple[str, str]:
    """
    Get specific language codes based on model type
    
    Args:
        language: Language name
        model_type: Model type
        
    Returns:
        Tuple of (source_language_code, target_language_code) for the model
    """
    if model_type.lower() == "nllb":
        # NLLB uses specific language codes
        if language.lower() in ["english", "en", "eng"]:
            return "eng_Latn", "eng_Latn"
        elif language.lower() in ["kinyarwanda", "kin", "rw"]:
            return "kin_Latn", "kin_Latn"
    else:
        # Default language codes
        if language.lower() in ["english", "en", "eng"]:
            return "en", "en"
        elif language.lower() in ["kinyarwanda", "kin", "rw"]:
            return "kin", "kin"
    
    # Default fallback
    return language.lower(), language.lower()

def detect_model_type(model_path: str) -> str:
    """
    Detect translation model type from path
    
    Args:
        model_path: Model path or name
        
    Returns:
        Model type string
    """
    if "nllb" in model_path.lower():
        return "nllb"
    elif "marian" in model_path.lower():
        return "marian"
    else:
        return "unknown"

def clear_translation_model_cache() -> int:
    """
    Clear translation model cache to free memory
    
    Returns:
        Number of models cleared from cache
    """
    count = len(TRANSLATION_MODEL_CACHE)
    TRANSLATION_MODEL_CACHE.clear()
    return count