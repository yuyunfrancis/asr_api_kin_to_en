from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import os

from app import config
from app.api.routes.transcription import router as transcription_router
from app.api.routes.translation import router as translation_router
from app.api.routes.captions import router as captions_router
from app.api.routes.process import router as process_router
from app.api.routes.jobs import router as jobs_router

# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
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
        "docs_url": "/docs"
    }

# API status endpoint
@app.get("/api/status")
async def status():
    gpu_status = "available" if config.gpu_available else "not available"
    models_directory = os.path.exists(config.MODELS_CACHE_DIR)
    
    return {
        "status": "operational",
        "gpu": gpu_status,
        "models_directory": models_directory,
        "version": config.API_VERSION
    }

# Handle application startup
@app.on_event("startup")
async def startup_event():
    # Create temp directory if it doesn't exist
    os.makedirs(config.TEMP_FILE_DIR, exist_ok=True)
    # Create models cache directory if it doesn't exist
    os.makedirs(config.MODELS_CACHE_DIR, exist_ok=True)
    
# Handle application shutdown
@app.on_event("shutdown")
async def shutdown_event():
    # Cleanup tasks could be added here
    pass

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)