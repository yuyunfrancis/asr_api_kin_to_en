from fastapi import APIRouter, BackgroundTasks, HTTPException, Body, Path, Query
from typing import List, Dict, Any, Optional

from app.core.schemas.requests import TranslationRequest, LanguageEnum
from app.core.services.translation import translate_text, translate_chunks
from app.core.services.job_storage import create_job, update_job, get_job, serialize_job

router = APIRouter()

@router.post("/translate", tags=["Translation"])
async def translate(
    background_tasks: BackgroundTasks,
    request: TranslationRequest,
    async_processing: bool = Query(False, description="Process translation asynchronously")
):
    """
    Translate text between languages
    """
    # Create job record
    job = create_job(
        job_type="translation",
        metadata={
            "source_language": request.source_language,
            "target_language": request.target_language,
            "text_length": len(request.text)
        }
    )
    
    # Handle synchronous vs asynchronous processing
    if async_processing:
        # Start translation in background
        background_tasks.add_task(
            process_translation,
            job["job_id"],
            request.text,
            request.source_language,
            request.target_language
        )
        
        return serialize_job(job)
    else:
        # Process synchronously
        await process_translation(
            job["job_id"],
            request.text,
            request.source_language,
            request.target_language
        )
        
        # Get updated job
        updated_job = get_job(job["job_id"])
        
        return serialize_job(updated_job)

@router.post("/jobs/{job_id}/translate", tags=["Translation"])
async def translate_job_results(
    background_tasks: BackgroundTasks,
    job_id: str = Path(..., description="Source transcription job ID"),
    target_language: LanguageEnum = Body(..., description="Target language for translation")
):
    """
    Translate results from a previous transcription job
    """
    # Get the source job
    source_job = get_job(job_id)
    
    if not source_job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if source_job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed")
    
    if "transcription" not in source_job["result"]:
        raise HTTPException(status_code=400, detail=f"Job {job_id} does not contain transcription results")
    
    # Extract source language and transcription chunks
    transcription = source_job["result"]["transcription"]
    source_language = transcription["language"]
    chunks = transcription["chunks"]
    
    # Create new translation job
    translation_job = create_job(
        job_type="translation",
        metadata={
            "source_job_id": job_id,
            "source_language": source_language,
            "target_language": target_language,
            "chunk_count": len(chunks)
        }
    )
    
    # Start translation in background
    background_tasks.add_task(
        process_job_translation,
        translation_job["job_id"],
        chunks,
        source_language,
        target_language
    )
    
    return serialize_job(translation_job)

async def process_translation(
    job_id: str,
    text: str,
    source_language: str,
    target_language: str
):
    """
    Process translation in background
    """
    try:
        # Update job status
        update_job(
            job_id=job_id,
            status="processing",
            progress=10,
            message="Starting translation"
        )
        
        # Translate text
        translated_text, processing_time = await translate_text(
            text,
            source_language,
            target_language
        )
        
        # Format results
        result = {
            "source_language": source_language,
            "target_language": target_language,
            "original_text": text,
            "translated_text": translated_text,
            "processing_time": processing_time
        }
        
        # Update job status
        update_job(
            job_id=job_id,
            status="completed",
            progress=100,
            message="Translation completed successfully",
            result={
                "translation": result
            }
        )
        
    except Exception as e:
        # Update job status on error
        update_job(
            job_id=job_id,
            status="failed",
            message=f"Translation failed: {str(e)}"
        )

async def process_job_translation(
    job_id: str,
    chunks: List[Dict[str, Any]],
    source_language: str,
    target_language: str
):
    """
    Process translation of transcription chunks in background
    """
    try:
        # Update job status
        update_job(
            job_id=job_id,
            status="processing",
            progress=10,
            message="Starting translation of transcription chunks"
        )
        
        # Extract full text for complete translation
        full_text = " ".join([chunk["text"] for chunk in chunks])
        
        # Translate full text
        translated_text, full_text_time = await translate_text(
            full_text,
            source_language,
            target_language
        )
        
        # Update progress
        update_job(
            job_id=job_id,
            progress=50,
            message="Full text translated, processing chunks"
        )
        
        # Translate individual chunks
        translated_chunks, chunk_time = await translate_chunks(
            chunks,
            source_language,
            target_language
        )
        
        # Calculate total processing time
        total_time = full_text_time + chunk_time
        
        # Format results
        result = {
            "source_language": source_language,
            "target_language": target_language,
            "full_text": translated_text,
            "chunks": translated_chunks,
            "processing_time": total_time
        }
        
        # Update job status
        update_job(
            job_id=job_id,
            status="completed",
            progress=100,
            message="Translation completed successfully",
            result={
                "translation": result
            }
        )
        
    except Exception as e:
        # Update job status on error
        update_job(
            job_id=job_id,
            status="failed",
            message=f"Translation failed: {str(e)}"
        )

@router.get("/models/translation", tags=["Translation"])
async def list_translation_models():
    """
    List available translation models
    """
    models = [
        {
            "id": "kin-to-en",
            "name": "Kinyarwanda to English",
            "source_language": "kinyarwanda",
            "target_language": "english",
            "description": "Marian model fine-tuned for Kinyarwanda to English translation"
        },
        {
            "id": "en-to-kin",
            "name": "English to Kinyarwanda",
            "source_language": "english",
            "target_language": "kinyarwanda",
            "description": "NLLB model fine-tuned for English to Kinyarwanda translation"
        }
    ]
    
    return {"models": models}