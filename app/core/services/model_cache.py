import os
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
import torch
import redis
import json
from collections import OrderedDict

from app import config

# Configure logging
logger = logging.getLogger("api.services.model_cache")

# Redis client for caching model metadata
redis_client = redis.Redis(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    db=config.REDIS_CACHE_DB,
    decode_responses=True
)

# In-memory LRU cache for loaded models
MODEL_CACHE = OrderedDict()

# Maximum number of models to keep in memory
MAX_CACHE_SIZE = config.MAX_MODEL_CACHE_SIZE

# Redis keys
MODEL_INFO_PREFIX = "model:info:"
MODEL_USAGE_PREFIX = "model:usage:"
MODEL_LIST_KEY = "models:list"

def register_model(model_id: str, model_type: str, language: str, description: str) -> Dict[str, Any]:
    """
    Register a model in the cache system
    
    Args:
        model_id: Unique model identifier (e.g., "whisper-small")
        model_type: Type of model (e.g., "transcription", "translation")
        language: Language the model supports
        description: Brief description of the model
        
    Returns:
        Model metadata
    """
    model_info = {
        "model_id": model_id,
        "model_type": model_type,
        "language": language,
        "description": description,
        "registered_at": time.time(),
        "last_used": None,
        "use_count": 0
    }
    
    # Store model info in Redis
    redis_client.set(f"{MODEL_INFO_PREFIX}{model_id}", json.dumps(model_info))
    
    # Add to model list
    redis_client.sadd(MODEL_LIST_KEY, model_id)
    
    logger.info(f"Registered model: {model_id}")
    return model_info

def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Get model metadata from cache
    
    Args:
        model_id: Unique model identifier
        
    Returns:
        Model metadata if found, None otherwise
    """
    model_info_data = redis_client.get(f"{MODEL_INFO_PREFIX}{model_id}")
    
    if not model_info_data:
        return None
        
    try:
        return json.loads(model_info_data)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode model info for {model_id}")
        return None

def update_model_usage(model_id: str) -> None:
    """
    Update model usage statistics
    
    Args:
        model_id: Unique model identifier
    """
    now = time.time()
    
    # Update usage counter
    redis_client.zincrby(f"{MODEL_USAGE_PREFIX}count", 1, model_id)
    
    # Update last used timestamp
    redis_client.zadd(f"{MODEL_USAGE_PREFIX}timestamp", {model_id: now})
    
    # Update model info
    model_info = get_model_info(model_id)
    if model_info:
        model_info["last_used"] = now
        model_info["use_count"] += 1
        redis_client.set(f"{MODEL_INFO_PREFIX}{model_id}", json.dumps(model_info))

def load_model(model_id: str, loader_func, *args, **kwargs) -> Any:
    """
    Load a model, using cache if available
    
    Args:
        model_id: Unique model identifier
        loader_func: Function to load the model if not in cache
        *args, **kwargs: Arguments to pass to loader_func
        
    Returns:
        Loaded model
    """
    # Check if model is in memory cache
    if model_id in MODEL_CACHE:
        logger.info(f"Using cached model: {model_id}")
        
        # Move model to the end of OrderedDict (most recently used)
        model = MODEL_CACHE.pop(model_id)
        MODEL_CACHE[model_id] = model
        
        # Update usage statistics
        update_model_usage(model_id)
        
        return model
    
    # Model not in cache, load it
    logger.info(f"Loading model: {model_id}")
    
    try:
        # Load the model using the provided function
        model = loader_func(*args, **kwargs)
        
        # Add to cache
        MODEL_CACHE[model_id] = model
        
        # If cache is full, remove least recently used model
        if len(MODEL_CACHE) > MAX_CACHE_SIZE:
            # Get least recently used model
            oldest_model_id, _ = next(iter(MODEL_CACHE.items()))
            
            # Remove it from cache
            MODEL_CACHE.pop(oldest_model_id)
            logger.info(f"Removed least recently used model from cache: {oldest_model_id}")
        
        # Update usage statistics
        update_model_usage(model_id)
        
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        raise

def get_popular_models(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get most frequently used models
    
    Args:
        limit: Maximum number of models to return
        
    Returns:
        List of model metadata
    """
    # Get most frequently used models
    popular_model_ids = redis_client.zrevrange(f"{MODEL_USAGE_PREFIX}count", 0, limit - 1)
    
    # Get metadata for each model
    models = []
    for model_id in popular_model_ids:
        model_info = get_model_info(model_id)
        if model_info:
            models.append(model_info)
    
    return models

def get_recent_models(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get most recently used models
    
    Args:
        limit: Maximum number of models to return
        
    Returns:
        List of model metadata
    """
    # Get most recently used models
    recent_model_ids = redis_client.zrevrange(f"{MODEL_USAGE_PREFIX}timestamp", 0, limit - 1)
    
    # Get metadata for each model
    models = []
    for model_id in recent_model_ids:
        model_info = get_model_info(model_id)
        if model_info:
            models.append(model_info)
    
    return models

def clear_model_cache() -> int:
    """
    Clear the model cache
    
    Returns:
        Number of models cleared from cache
    """
    count = len(MODEL_CACHE)
    MODEL_CACHE.clear()
    return count