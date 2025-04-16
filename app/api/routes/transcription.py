from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import time
import os
import uuid
import shutil

from app import config
from app.core.schemas.requests import TranscriptionRequest, LanguageEnum, ModelTypeEnum
from app.core.services.transcription import (
    transcribe_with_pipeline,
    transcribe_with_whisper_model,
    transcribe_with_production_approach
)
from app.core.services.job_storage import create_job, update_job, serialize_job
from app.utils.file_utils import save_upload_file, clean_temp_file
from app.workers.tasks import process_transcription

router = APIRouter()

@router.post("/transcribe", tags=["Transcription"])
async def transcribe_audio(
    file: UploadFile = File(...),
    language: LanguageEnum = Form(LanguageEnum.english),
    model_type: ModelTypeEnum = Form(ModelTypeEnum.whisper),
    chunk_size: int = Form(10),
    overlap_size: int = Form(5),
    use_mfa: bool = Form(False),
    async_processing: bool = Form(True)
):
    """
    Transcribe audio file
    """
    # Save uploaded file
    file_path = await save_upload_file(file)
    
    # Create job record
    job = create_job(
        job_type="transcription",
        metadata={
            "file_name": file.filename,
            "language": language,
            "model_type": model_type,
            "chunk_size": chunk_size,
            "overlap_size": overlap_size,
            "use_mfa": use_mfa
        }
    )
    
    # Start transcription task
    process_transcription.delay(
        job_id=job["job_id"],
        file_path=file_path,
        language=language,
        model_type=model_type,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        use_mfa=use_mfa
    )
    
    return serialize_job(job)

async def process_transcription(
    job_id: str,
    file_path: str,
    language: str,
    model_type: str,
    chunk_size: int,
    overlap_size: int,
    use_mfa: bool
):
    """
    Process transcription in background
    """
    try:
        # Update job status
        update_job(
            job_id=job_id,
            status="processing",
            progress=10,
            message="Starting transcription"
        )
        
        # Get language code
        language_code = config.LANGUAGE_CODES.get(language.lower(), "en")
        
        # Process based on model type
        if model_type == "nemo":
            # Use production approach for Kinyarwanda
            update_job(
                job_id=job_id,
                status="processing",
                progress=20,
                message="Transcribing with NeMo hybrid approach"
            )
            
            chunks, processing_time, error = await transcribe_with_production_approach(
                audio_path=file_path,
                language=language_code,
                use_mfa=use_mfa
            )
            
            if error:
                update_job(
                    job_id=job_id,
                    status="processing",
                    progress=80,
                    message=f"Transcription completed with warnings: {error}"
                )
            
        elif model_type == "whisper-kinyarwanda":
            # Use Whisper model fine-tuned for Kinyarwanda
            update_job(
                job_id=job_id,
                status="processing",
                progress=20,
                message="Transcribing with Kinyarwanda-specific Whisper model"
            )
            
            chunks, processing_time = await transcribe_with_whisper_model(
                audio_path=file_path,
                language=language_code,
                chunk_size_seconds=chunk_size,
                overlap_seconds=overlap_size
            )
            
        else:
            # Use general Whisper model
            update_job(
                job_id=job_id,
                status="processing",
                progress=20,
                message="Transcribing with Whisper model"
            )
            
            chunks, processing_time = await transcribe_with_pipeline(
                audio_path=file_path,
                language=language_code
            )
        
        # Extract full text
        full_text = " ".join([chunk["text"] for chunk in chunks])
        
        # Calculate duration from last chunk end time
        duration = chunks[-1]["end_time"] if chunks else 0
        
        # Format results
        result = {
            "language": language,
            "model_used": model_type,
            "full_text": full_text,
            "chunks": chunks,
            "duration": duration,
            "processing_time": processing_time
        }
        
        # Update job status
        update_job(
            job_id=job_id,
            status="completed",
            progress=100,
            message="Transcription completed successfully",
            result={
                "transcription": result
            }
        )
        
    except Exception as e:
        # Update job status on error
        update_job(
            job_id=job_id,
            status="failed",
            message=f"Transcription failed: {str(e)}"
        )
        
    finally:
        # Clean up temporary file
        clean_temp_file(file_path)

@router.get("/models/transcription", tags=["Transcription"])
async def list_transcription_models():
    """
    List available transcription models
    """
    models = [
        {
            "id": "whisper",
            "name": "Whisper Small",
            "language": "English",
            "description": "OpenAI Whisper Small model for English transcription"
        },
        {
            "id": "whisper-kinyarwanda",
            "name": "Whisper Kinyarwanda",
            "language": "Kinyarwanda",
            "description": "Whisper Small model fine-tuned for Kinyarwanda"
        },
        {
            "id": "nemo",
            "name": "NeMo Kinyarwanda",
            "language": "Kinyarwanda",
            "description": "NVIDIA NeMo-based model with high accuracy for Kinyarwanda"
        }
    ]
    
    return {"models": models}