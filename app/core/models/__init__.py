import os
import logging
import torch
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from typing import Dict, Any, Tuple, Optional, Union
import subprocess
import sys

from app import config

# Configure logging
logger = logging.getLogger("api.models")

# Cache for loaded models
MODEL_CACHE = {}

def load_pipeline_model(model_name: str = "openai/whisper-small") -> Any:
    """
    Load model using the pipeline approach
    
    Args:
        model_name: The model name/path
        
    Returns:
        The loaded pipeline
    """
    cache_key = f"pipeline_{model_name}"
    
    # Check if model is in cache
    if cache_key in MODEL_CACHE:
        logger.info(f"Using cached pipeline model: {model_name}")
        return MODEL_CACHE[cache_key]
    
    logger.info(f"Loading pipeline model: {model_name}")
    device = 'cuda' if config.gpu_available else 'cpu'
    
    pipe = pipeline(
        "automatic-speech-recognition", 
        model_name, 
        chunk_length_s=30, 
        stride_length_s=5, 
        return_timestamps=True, 
        device=device,
        framework="pt"
    )
    
    # Cache the model
    MODEL_CACHE[cache_key] = pipe
    
    return pipe

def load_whisper_model(model_name: str = "mbazaNLP/Whisper-Small-Kinyarwanda") -> Tuple[Any, Any]:
    """
    Load Whisper model and processor
    
    Args:
        model_name: The model name/path
        
    Returns:
        Tuple containing the processor and model
    """
    cache_key = f"whisper_{model_name}"
    
    # Check if model is in cache
    if cache_key in MODEL_CACHE:
        logger.info(f"Using cached Whisper model: {model_name}")
        return MODEL_CACHE[cache_key]
    
    logger.info(f"Loading Whisper model: {model_name}")
    
    processor = WhisperProcessor.from_pretrained(model_name, token=config.HUGGING_FACE_TOKEN)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, token=config.HUGGING_FACE_TOKEN)
    
    # Move to GPU if available
    if config.gpu_available:
        model = model.to("cuda")
        logger.info("Using GPU for transcription")
    
    # Cache the model
    MODEL_CACHE[cache_key] = (processor, model)
    
    return processor, model

def load_nemo_model(model_name: str = "mbazaNLP/Kinyarwanda_nemo_stt_conformer_model") -> Tuple[Any, bool, Optional[str]]:
    """
    Load NVIDIA NeMo ASR model
    
    Args:
        model_name: The model name/path
        
    Returns:
        Tuple containing the model, success flag, and error message (if any)
    """
    cache_key = f"nemo_{model_name}"
    
    # Check if model is in cache
    if cache_key in MODEL_CACHE:
        logger.info(f"Using cached NeMo model: {model_name}")
        return MODEL_CACHE[cache_key][0], MODEL_CACHE[cache_key][1], None
    
    logger.info(f"Loading NeMo model: {model_name}")
    
    try:
        # Check if nemo_toolkit is installed
        import importlib
        nemo_spec = importlib.util.find_spec("nemo_toolkit")
        nemo_asr_spec = importlib.util.find_spec("nemo.collections.asr")
        
        if nemo_spec is None or nemo_asr_spec is None:
            logger.warning("NeMo toolkit not found. Attempting to install...")
            # Try to install NeMo toolkit
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nemo_toolkit[all]"])
            
        # Import NeMo after ensuring it's installed
        import nemo.collections.asr as nemo_asr
        model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        
        # Move to GPU if available
        if config.gpu_available:
            model = model.cuda()
            logger.info("Using GPU for NeMo model")
        
        # Cache the model
        MODEL_CACHE[cache_key] = (model, True)
        
        return model, True, None
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "access" in error_msg.lower() or "restricted" in error_msg.lower():
            logger.error(f"Authentication error: No access to model {model_name}. Please check your Hugging Face token.")
            error_message = f"Authentication error: You don't have access to the NeMo model. Please check your Hugging Face token or request access to the model."
            return None, False, error_message
        else:
            logger.error(f"Failed to load NeMo model: {e}")
            return None, False, str(e)

def initialize_transcription_models(model_type: str, language_code: str) -> Dict[str, Any]:
    """
    Initialize the necessary models based on selection
    
    Args:
        model_type: Type of model to initialize
        language_code: Language code
        
    Returns:
        Dictionary containing initialized models
    """
    if "nemo" in model_type:
        # Load NeMo model
        nemo_model, nemo_loaded, error = load_nemo_model("mbazaNLP/Kinyarwanda_nemo_stt_conformer_model")
        
        # Always load Whisper as well for hybrid approach
        whisper_processor, whisper_model = load_whisper_model("mbazaNLP/Whisper-Small-Kinyarwanda")
        
        return {
            "nemo_model": nemo_model if nemo_loaded else None,
            "whisper_processor": whisper_processor,
            "whisper_model": whisper_model,
            "nemo_loaded": nemo_loaded,
            "error": error
        }
    else:
        # Load Whisper only
        if "kinyarwanda" in model_type:
            processor, model = load_whisper_model("mbazaNLP/Whisper-Small-Kinyarwanda")
            return {
                "nemo_model": None,
                "whisper_processor": processor,
                "whisper_model": model,
                "nemo_loaded": False,
                "error": None
            }
        else:
            model = load_pipeline_model("openai/whisper-small")
            return {
                "nemo_model": None,
                "whisper_model": model,
                "whisper_processor": None,
                "nemo_loaded": False,
                "error": None
            }

def clear_model_cache() -> int:
    """
    Clear model cache to free memory
    
    Returns:
        Number of models cleared from cache
    """
    count = len(MODEL_CACHE)
    MODEL_CACHE.clear()
    return count