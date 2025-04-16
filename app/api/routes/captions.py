from fastapi import APIRouter, BackgroundTasks, HTTPException, Body, Path, Query, Depends
from fastapi.responses import FileResponse
from typing import List, Dict, Any, Optional
import os

from app.core.services.captions import (
    generate_srt_captions, 
    generate_vtt_captions,
    save_caption_files
)
from app.core.services.job_storage import create_job, update_job, get_job, serialize_job
from app import config
from app.workers.tasks import process_captions

router = APIRouter()

@router.post("/generate-captions", tags=["Captions"])
async def generate_captions(
    job_id: str = Body(..., description="Source job ID (transcription or translation)"),
    include_original: bool = Body(True, description="Include original text in captions"),
    include_translation: bool = Body(True, description="Include translated text in captions"),
    formats: List[str] = Body(["srt", "vtt"], description="Caption formats to generate")
):
    """
    Generate caption files from transcription or translation results
    """
    # Get the source job
    source_job = get_job(job_id)
    
    if not source_job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if source_job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed")
    
    # Determine if we have transcription or translation results
    chunks = None
    job_type = source_job["job_type"]
    
    if job_type == "transcription" and "transcription" in source_job["result"]:
        # Simple transcription without translation
        chunks = source_job["result"]["transcription"]["chunks"]
        has_translation = False
    elif job_type == "translation" and "translation" in source_job["result"] and "chunks" in source_job["result"]["translation"]:
        # Translation job with chunks (from transcription)
        chunks = source_job["result"]["translation"]["chunks"]
        has_translation = True
    else:
        raise HTTPException(status_code=400, detail=f"Job {job_id} does not contain suitable data for caption generation")
    
    # Create new caption job
    caption_job = create_job(
        job_type="captions",
        metadata={
            "source_job_id": job_id,
            "include_original": include_original,
            "include_translation": include_translation,
            "formats": formats,
            "has_translation": has_translation
        }
    )
    
    # Start caption generation task
    process_captions.delay(
        job_id=caption_job["job_id"],
        chunks=chunks,
        include_original=include_original,
        include_translation=include_translation and has_translation,
        formats=formats
    )
    
    return serialize_job(caption_job)
async def process_caption_generation(
    job_id: str,
    chunks: List[Dict[str, Any]],
    include_original: bool,
    include_translation: bool,
    formats: List[str]
):
    """
    Process caption generation in background
    """
    try:
        # Update job status
        update_job(
            job_id=job_id,
            status="processing",
            progress=10,
            message="Starting caption generation"
        )
        
        result = {
            "formats": {}
        }
        
        # Generate SRT if requested
        if "srt" in formats:
            update_job(
                job_id=job_id,
                progress=30,
                message="Generating SRT captions"
            )
            
            srt_content, srt_time = await generate_srt_captions(
                chunks,
                include_original,
                include_translation
            )
            
            # Save SRT file
            from app.utils.file_utils import save_caption_file
            srt_path = save_caption_file(srt_content, job_id, "srt")
            
            # Add to result
            result["formats"]["srt"] = {
                "url": f"/api/files/captions/{job_id}/srt",
                "preview": srt_content[:500] + ("..." if len(srt_content) > 500 else "")
            }
        
        # Generate VTT if requested
        if "vtt" in formats:
            update_job(
                job_id=job_id,
                progress=60,
                message="Generating VTT captions"
            )
            
            vtt_content, vtt_time = await generate_vtt_captions(
                chunks,
                include_original,
                include_translation
            )
            
            # Save VTT file
            from app.utils.file_utils import save_caption_file
            vtt_path = save_caption_file(vtt_content, job_id, "vtt")
            
            # Add to result
            result["formats"]["vtt"] = {
                "url": f"/api/files/captions/{job_id}/vtt",
                "preview": vtt_content[:500] + ("..." if len(vtt_content) > 500 else "")
            }
        
        # Add metadata
        result["metadata"] = {
            "word_count": sum(len(chunk.get("text", "").split()) for chunk in chunks),
            "line_count": len(chunks),
            "include_original": include_original,
            "include_translation": include_translation
        }
        
        # Update job status
        update_job(
            job_id=job_id,
            status="completed",
            progress=100,
            message="Caption generation completed successfully",
            result={
                "captions": result
            }
        )
        
    except Exception as e:
        # Update job status on error
        update_job(
            job_id=job_id,
            status="failed",
            message=f"Caption generation failed: {str(e)}"
        )

@router.get("/files/captions/{job_id}/{format_type}", tags=["Captions"])
async def get_caption_file(
    job_id: str = Path(..., description="Caption job ID"),
    format_type: str = Path(..., description="Caption format (srt or vtt)")
):
    """
    Get generated caption file
    """
    # Check if job exists
    job = get_job(job_id)
    
    if not job or job["status"] != "completed":
        raise HTTPException(status_code=404, detail=f"Caption job {job_id} not found or not completed")
    
    # Determine file path
    file_path = os.path.join(config.TEMP_FILE_DIR, "captions", job_id, f"captions.{format_type}")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Caption file not found")
    
    # Set content type based on format
    media_type = "text/srt" if format_type == "srt" else "text/vtt"
    
    # Return file
    return FileResponse(
        file_path, 
        media_type=media_type,
        filename=f"captions.{format_type}"
    )