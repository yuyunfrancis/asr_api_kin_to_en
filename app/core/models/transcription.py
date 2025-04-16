import os
import logging
import torch
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from typing import Dict, Any, Tuple, Optional, Union
import subprocess
import sys

from app import config
from app.core.services.model_cache import load_model, register_model, get_model_info

# Configure logging
logger = logging.getLogger("api.models")

def load_pipeline_model(model_name: str = "openai/whisper-small") -> Any:
    """
    Load model using the pipeline approach with caching
    
    Args:
        model_name: The model name/path
        
    Returns:
        The loaded pipeline
    """
    cache_key = f"pipeline_{model_name}"
    
    # Define loader function
    def loader_func():
        device = 'cuda' if config.gpu_available else 'cpu'
        
        return pipeline(
            "automatic-speech-recognition", 
            model_name, 
            chunk_length_s=30, 
            stride_length_s=5, 
            return_timestamps=True, 
            device=device,
            framework="pt"
        )
    
    # Register model if not already registered
    if not get_model_info(cache_key):
        register_model(
            model_id=cache_key,
            model_type="transcription",
            language="en" if "whisper-small" in model_name else "kin",
            description=f"Pipeline model: {model_name}"
        )
    
    # Load model with caching
    return load_model(cache_key, loader_func)

def load_whisper_model(model_name: str = "mbazaNLP/Whisper-Small-Kinyarwanda") -> Tuple[Any, Any]:
    """
    Load Whisper model and processor with caching
    
    Args:
        model_name: The model name/path
        
    Returns:
        Tuple containing the processor and model
    """
    cache_key = f"whisper_{model_name}"
    
    # Define loader function
    def loader_func():
        processor = WhisperProcessor.from_pretrained(model_name, token=config.HUGGING_FACE_TOKEN)
        model = WhisperForConditionalGeneration.from_pretrained(model_name, token=config.HUGGING_FACE_TOKEN)
        
        # Move to GPU if available
        if config.gpu_available:
            model = model.to("cuda")
            logger.info("Using GPU for transcription")
        
        return (processor, model)
    
    # Register model if not already registered
    if not get_model_info(cache_key):
        register_model(
            model_id=cache_key,
            model_type="transcription",
            language="en" if "whisper-small" in model_name else "kin",
            description=f"Whisper model: {model_name}"
        )
    
    # Load model with caching
    return load_model(cache_key, loader_func)

def load_nemo_model(model_name: str = "mbazaNLP/Kinyarwanda_nemo_stt_conformer_model") -> Tuple[Any, bool, Optional[str]]:
    """
    Load NVIDIA NeMo ASR model with graceful fallback
    """
    cache_key = f"nemo_{model_name}"
    
    # First check if NeMo is available without attempting installation
    try:
        import importlib.util
        if importlib.util.find_spec("nemo_toolkit") is None and importlib.util.find_spec("nemo.collections.asr") is None:
            logger.warning("NeMo toolkit is not installed. Skipping NeMo model loading.")
            return None, False, "NeMo toolkit is not installed. Using Whisper model instead."
    except ImportError:
        logger.warning("NeMo toolkit is not installed. Skipping NeMo model loading.")
        return None, False, "NeMo toolkit is not installed. Using Whisper model instead."
    
    # Define loader function
    def loader_func():
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
            
            return (model, True)
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "access" in error_msg.lower() or "restricted" in error_msg.lower():
                logger.error(f"Authentication error: No access to model {model_name}. Please check your Hugging Face token.")
                error_message = f"Authentication error: You don't have access to the NeMo model. Please check your Hugging Face token or request access to the model."
                return (None, False)
            else:
                logger.error(f"Failed to load NeMo model: {e}")
                return (None, False)
    
    # Register model if not already registered
    if not get_model_info(cache_key):
        register_model(
            model_id=cache_key,
            model_type="transcription",
            language="kin",
            description=f"NeMo ASR model: {model_name}"
        )
    
    try:
        # Load model with caching
        model, success = load_model(cache_key, loader_func)
        return model, success, None if success else "Failed to load NeMo model"
    except Exception as e:
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
    from app.core.services.model_cache import clear_model_cache as clear_cache
    return clear_cache()