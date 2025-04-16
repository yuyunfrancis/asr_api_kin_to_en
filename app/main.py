from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import os
import logging
import torch

from app import config
from app.api.routes.transcription import router as transcription_router
from app.api.routes.translation import router as translation_router
from app.api.routes.captions import router as captions_router
from app.api.routes.process import router as process_router
from app.api.routes.jobs import router as jobs_router

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include routers
app.include_router(process_router, prefix="/api", tags=["Process"])
app.include_router(transcription_router, prefix="/api", tags=["Transcription"])
app.include_router(translation_router, prefix="/api", tags=["Translation"])
app.include_router(captions_router, prefix="/api", tags=["Captions"])
app.include_router(jobs_router, prefix="/api", tags=["Jobs"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Audio Transcription & Translation API",
        "status": "online",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

# API status endpoint
@app.get("/api/status")
async def status():
    # Check GPU status
    gpu_status = {
        "available": config.gpu_available,
        "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }
    
    # Check model directory
    models_directory = {
        "exists": os.path.exists(config.MODELS_CACHE_DIR),
        "path": os.path.abspath(config.MODELS_CACHE_DIR),
        "size_mb": sum(os.path.getsize(os.path.join(config.MODELS_CACHE_DIR, f)) 
                    for f in os.listdir(config.MODELS_CACHE_DIR) 
                    if os.path.isfile(os.path.join(config.MODELS_CACHE_DIR, f))) / (1024 * 1024)
                    if os.path.exists(config.MODELS_CACHE_DIR) else 0
    }
    
    # Check Redis connection
    redis_status = "connected"
    try:
        from app.core.services.redis_job_storage import redis_client
        redis_client.ping()
    except Exception as e:
        redis_status = f"disconnected: {str(e)}"
    
    # Get active models
    models_info = []
    try:
        from app.core.services.model_cache import get_recent_models
        models_info = get_recent_models(limit=5)
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
    
    return {
        "status": "operational",
        "version": config.API_VERSION,
        "gpu": gpu_status,
        "models_directory": models_directory,
        "redis": redis_status,
        "recent_models": models_info
    }

# Handle application startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Audio Transcription & Translation API...")
    
    # Create directories if they don't exist
    os.makedirs(config.TEMP_FILE_DIR, exist_ok=True)
    os.makedirs(config.MODELS_CACHE_DIR, exist_ok=True)
    
    # Log GPU status
    if torch.cuda.is_available() and config.USE_GPU:
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} CUDA device(s):")
        for i in range(gpu_count):
            logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"GPU acceleration is enabled")
    else:
        logger.warning("GPU acceleration is not available or disabled")
    
    # Initialize Redis connection
    try:
        from app.core.services.redis_job_storage import redis_client
        redis_client.ping()
        logger.info("Redis connection successful")
        
        # Register default models
        try:
            from app.core.services.model_cache import register_model, get_model_info
            
            # Register transcription models
            if not get_model_info("pipeline_openai/whisper-small"):
                register_model(
                    model_id="pipeline_openai/whisper-small",
                    model_type="transcription",
                    language="en",
                    description="OpenAI Whisper Small model for English transcription"
                )
            
            if not get_model_info("whisper_mbazaNLP/Whisper-Small-Kinyarwanda"):
                register_model(
                    model_id="whisper_mbazaNLP/Whisper-Small-Kinyarwanda",
                    model_type="transcription",
                    language="kin",
                    description="Whisper Small model fine-tuned for Kinyarwanda"
                )
            
            if not get_model_info("nemo_mbazaNLP/Kinyarwanda_nemo_stt_conformer_model"):
                register_model(
                    model_id="nemo_mbazaNLP/Kinyarwanda_nemo_stt_conformer_model",
                    model_type="transcription",
                    language="kin",
                    description="NVIDIA NeMo-based model with high accuracy for Kinyarwanda"
                )
            
            # Register translation models
            if not get_model_info("translation_kin-to-en"):
                register_model(
                    model_id="translation_kin-to-en",
                    model_type="translation",
                    language="kin-en",
                    description="Kinyarwanda to English translation model"
                )
            
            if not get_model_info("translation_en-to-kin"):
                register_model(
                    model_id="translation_en-to-kin",
                    model_type="translation",
                    language="en-kin",
                    description="English to Kinyarwanda translation model"
                )
                
            logger.info("Default models registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to register models: {e}")
        
        # Initialize Celery (if available)
        try:
            from app.workers.celery_app import app as celery_app
            logger.info("Celery configuration loaded")
        except ImportError:
            logger.warning("Celery not available - background tasks will run in-process")
        except Exception as e:
            logger.error(f"Failed to initialize Celery: {e}")
        
    except Exception as e:
        logger.error(f"Redis initialization failed: {e}")
        logger.warning("Falling back to in-memory job storage")

# Handle application shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API...")
    
    # Clean up resources
    try:
        from app.core.services.model_cache import clear_model_cache
        cleared_count = clear_model_cache()
        logger.info(f"Cleared {cleared_count} models from cache")
    except Exception as e:
        logger.error(f"Error clearing model cache: {e}")
    
    # Clean up temporary files
    try:
        from app.utils.file_utils import cleanup_old_files
        cleaned_count = cleanup_old_files(0)  # Clean all files on shutdown
        logger.info(f"Cleaned {cleaned_count} temporary files")
    except Exception as e:
        logger.error(f"Error cleaning temporary files: {e}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)