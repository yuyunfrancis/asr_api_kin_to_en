from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import time
import json
import os
import logging as logger

from app import config
from app.core.schemas.requests import LanguageEnum, ModelTypeEnum, TranslationOptions, CaptionOptions
from app.core.services.job_storage import create_job, update_job, get_job, serialize_job
from app.utils.file_utils import save_upload_file, clean_temp_file

router = APIRouter()

@router.post("/process-audio", tags=["Process"])
async def process_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_language: LanguageEnum = Form(LanguageEnum.english),
    transcription_model: ModelTypeEnum = Form(ModelTypeEnum.whisper),
    transcription_options: Optional[str] = Form(None),
    translation_options: Optional[str] = Form(None),
    caption_options: Optional[str] = Form(None),
    processing_mode: str = Form("async")
):
    """
    Process audio through the complete pipeline: transcription → translation → captions
    """
    # Parse JSON options or use defaults
    trans_options = {}
    if transcription_options:
        try:
            trans_options = json.loads(transcription_options)
        except:
            raise HTTPException(status_code=400, detail="Invalid transcription options JSON")
    
    # Default transcription options
    chunk_size = trans_options.get("chunk_size", 10)
    overlap_size = trans_options.get("overlap_size", 5)
    use_mfa = trans_options.get("use_mfa", False)
    
    # Parse translation options
    translate = False
    target_language = None
    translation_model = None
    
    if translation_options:
        try:
            trans_opt = json.loads(translation_options)
            translate = trans_opt.get("enabled", False)
            target_language = trans_opt.get("target_language")
            translation_model = trans_opt.get("model")
        except:
            raise HTTPException(status_code=400, detail="Invalid translation options JSON")
    
    # Parse caption options
    generate_captions = False
    caption_formats = ["srt", "vtt"]
    include_original = True
    include_translation = True
    
    if caption_options:
        try:
            cap_opt = json.loads(caption_options)
            generate_captions = cap_opt.get("enabled", False)
            caption_formats = cap_opt.get("formats", ["srt", "vtt"])
            include_original = cap_opt.get("include_original", True)
            include_translation = cap_opt.get("include_translation", True)
        except:
            raise HTTPException(status_code=400, detail="Invalid caption options JSON")
    
    # Save uploaded file
    file_path = await save_upload_file(file)
    
    # Create job record
    job = create_job(
        job_type="pipeline",
        metadata={
            "file_name": file.filename,
            "source_language": source_language,
            "transcription_model": transcription_model,
            "translate": translate,
            "target_language": target_language,
            "generate_captions": generate_captions,
            "pipeline_steps": ["transcription"] + 
                              (["translation"] if translate else []) + 
                              (["captions"] if generate_captions else [])
        }
    )
    
    # Start processing in background
    background_tasks.add_task(
        process_full_pipeline,
        job["job_id"],
        file_path,
        source_language,
        transcription_model,
        chunk_size,
        overlap_size,
        use_mfa,
        translate,
        target_language,
        translation_model,
        generate_captions,
        caption_formats,
        include_original,
        include_translation
    )
    
    return serialize_job(job)

async def process_full_pipeline(
    job_id: str,
    file_path: str,
    source_language: str,
    transcription_model: str,
    chunk_size: int,
    overlap_size: int,
    use_mfa: bool,
    translate: bool,
    target_language: Optional[str],
    translation_model: Optional[str],
    generate_captions: bool,
    caption_formats: List[str],
    include_original: bool,
    include_translation: bool
):
    """
    Process the complete pipeline in background
    """
    start_time = time.time()
    pipeline_results = {}
    
    try:
        # Step 1: Transcription
        update_job(
            job_id=job_id,
            status="processing",
            progress=10,
            message="Starting transcription"
        )
        
        # Get language code
        from app import config
        language_code = config.LANGUAGE_CODES.get(source_language.lower(), "en")
        
        # Import transcription functions
        from app.core.services.transcription import (
            transcribe_with_pipeline,
            transcribe_with_whisper_model,
            transcribe_with_production_approach
        )
        
        # Process based on model type
        if transcription_model == "nemo":
            # Use production approach for Kinyarwanda
            update_job(
                job_id=job_id,
                status="processing",
                progress=20,
                message="Transcribing with NeMo hybrid approach"
            )
            
            transcription_chunks, transcription_time, error = await transcribe_with_production_approach(
                audio_path=file_path,
                language=language_code,
                use_mfa=use_mfa
            )
            
        elif transcription_model == "whisper-kinyarwanda":
            # Use Whisper model fine-tuned for Kinyarwanda
            update_job(
                job_id=job_id,
                status="processing",
                progress=20,
                message="Transcribing with Kinyarwanda-specific Whisper model"
            )
            
            transcription_chunks, transcription_time = await transcribe_with_whisper_model(
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
            
            transcription_chunks, transcription_time = await transcribe_with_pipeline(
                audio_path=file_path,
                language=language_code
            )
        
        # Extract full text
        full_text = " ".join([chunk["text"] for chunk in transcription_chunks])
        
        # Calculate duration from last chunk end time
        duration = transcription_chunks[-1]["end_time"] if transcription_chunks else 0
        
        # Format transcription results
        transcription_result = {
            "language": source_language,
            "model_used": transcription_model,
            "full_text": full_text,
            "chunks": transcription_chunks,
            "duration": duration,
            "confidence_score": None  # Not available in this version
        }
        
        pipeline_results["transcription"] = transcription_result
        
        update_job(
            job_id=job_id,
            progress=40,
            message="Transcription completed successfully"
        )
        
        # Step 2: Translation (if requested)
        translation_chunks = None
        translation_time = 0
        
        if translate and target_language:
            update_job(
                job_id=job_id,
                progress=50,
                message=f"Starting translation to {target_language}"
            )
            
            # Import translation functions
            from app.core.services.translation import translate_text, translate_chunks
            
            # Translate full text
            translated_full_text, full_text_translation_time = await translate_text(
                full_text,
                source_language,
                target_language
            )
            
            # Translate individual chunks
            translation_chunks, chunk_translation_time = await translate_chunks(
                transcription_chunks,
                source_language,
                target_language
            )
            
            # Calculate total translation time
            translation_time = full_text_translation_time + chunk_translation_time
            
            # Format translation results
            if translate and target_language:
                translation_result = {
                    "language": target_language,
                    "model_used": translation_model,
                    "full_text": translated_full_text,
                    "chunks": translation_chunks,
                    "duration": duration
                }
            
            # Log the structure used for captions
            logger.info(f"Number of translation chunks for captions: {len(translation_chunks) if translation_chunks else 0}")
            if translation_chunks and len(translation_chunks) > 0:
                logger.info(f"First translation chunk: {translation_chunks[0]}")
            
            pipeline_results["translation"] = translation_result
            
            update_job(
                job_id=job_id,
                progress=70,
                message="Translation completed successfully"
            )
        
        # Step 3: Caption generation (if requested)
        caption_time = 0
        
        if generate_captions:
            update_job(
                job_id=job_id,
                progress=80,
                message="Generating captions"
            )
            
            # Import caption functions
            from app.core.services.captions import save_caption_files
            
            # Use translation chunks if available, otherwise use transcription chunks
            chunks_for_captions = translation_chunks if translation_chunks else transcription_chunks
            
            # Generate caption files
            caption_files, caption_time = await save_caption_files(
                chunks_for_captions,
                job_id,
                include_original,
                include_translation and translate  # Only include translation if it was performed
            )
            
            # Format caption results
            caption_result = {
                "formats": {}
            }
            
            # Add caption file URLs and previews
            for format_type, file_path in caption_files.items():
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                caption_result["formats"][format_type] = {
                    "url": f"/api/files/captions/{job_id}/{format_type}",
                    "download_valid_until": (time.time() + 86400),  # Valid for 24 hours
                    "preview": content[:500] + ("..." if len(content) > 500 else "")
                }
            
            # Add metadata
            caption_result["metadata"] = {
                "word_count": sum(len(chunk.get("text", "").split()) for chunk in chunks_for_captions),
                "line_count": len(chunks_for_captions),
                "duration": duration
            }
            
            pipeline_results["captions"] = caption_result
            
            update_job(
                job_id=job_id,
                progress=90,
                message="Caption generation completed successfully"
            )
        
        # Calculate total processing time
        total_processing_time = time.time() - start_time
        
        # Add processing stats
        pipeline_results["processing_stats"] = {
            "transcription_time": transcription_time,
            "translation_time": translation_time if translate else None,
            "caption_generation_time": caption_time if generate_captions else None,
            "total_processing_time": total_processing_time
        }
        
        # Update job with complete results
        update_job(
            job_id=job_id,
            status="completed",
            progress=100,
            message="Pipeline processing completed successfully",
            result={
                "pipeline_results": pipeline_results
            }
        )
    
    except Exception as e:
        # Update job status on error
        update_job(
            job_id=job_id,
            status="failed",
            message=f"Pipeline processing failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        clean_temp_file(file_path)
