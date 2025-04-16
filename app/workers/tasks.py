import logging
import time
from typing import Dict, Any, List, Optional

from app.workers.celery_app import app
from app.core.services.job_storage import update_job, get_job
from app.utils.file_utils import clean_temp_file, save_caption_file

# Configure logging
logger = logging.getLogger("celery.tasks")

# @app.task(bind=True, name="process_transcription")
def process_transcription(
    self,
    job_id: str,
    file_path: str,
    language: str,
    model_type: str,
    chunk_size: int,
    overlap_size: int,
    use_mfa: bool
):
    """
    Process transcription in background with Celery
    """
    try:
        # Update job status
        update_job(
            job_id=job_id,
            status="processing",
            progress=10,
            message="Starting transcription"
        )
        
        # Import here to avoid circular imports
        from app import config
        from app.core.services.transcription import (
            transcribe_with_pipeline,
            transcribe_with_whisper_model,
            transcribe_with_production_approach
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
            
            chunks, processing_time, error = transcribe_with_production_approach(
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
            
            chunks, processing_time = transcribe_with_whisper_model(
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
            
            chunks, processing_time = transcribe_with_pipeline(
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
        
        return {
            "job_id": job_id,
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        # Update job status on error
        update_job(
            job_id=job_id,
            status="failed",
            message=f"Transcription failed: {str(e)}"
        )
        
        # Re-raise for Celery to handle
        logger.error(f"Transcription task failed: {str(e)}")
        raise
        
    finally:
        # Clean up temporary file
        clean_temp_file(file_path)

@app.task(bind=True, name="process_translation")
def process_translation(
    self,
    job_id: str,
    chunks: List[Dict[str, Any]],
    source_language: str,
    target_language: str
):
    """
    Process translation of transcription chunks
    """
    try:
        # Update job status
        update_job(
            job_id=job_id,
            status="processing",
            progress=10,
            message=f"Starting translation from {source_language} to {target_language}"
        )
        
        # Import translation functions
        from app.core.services.translation import translate_text, translate_chunks
        
        # Extract full text for complete translation
        full_text = " ".join([chunk["text"] for chunk in chunks])
        
        # Update progress
        update_job(
            job_id=job_id,
            progress=30,
            message="Translating full text"
        )
        
        # Translate full text
        translated_text, full_text_time = translate_text(
            full_text,
            source_language,
            target_language
        )
        
        # Update progress
        update_job(
            job_id=job_id,
            progress=60,
            message="Translating individual chunks"
        )
        
        # Translate individual chunks
        translated_chunks, chunk_time = translate_chunks(
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
        
        return {
            "job_id": job_id,
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        # Update job status on error
        update_job(
            job_id=job_id,
            status="failed",
            message=f"Translation failed: {str(e)}"
        )
        
        # Re-raise for Celery to handle
        logger.error(f"Translation task failed: {str(e)}")
        raise

@app.task(bind=True, name="process_captions")
def process_captions(
    self,
    job_id: str,
    chunks: List[Dict[str, Any]],
    include_original: bool,
    include_translation: bool,
    formats: List[str]
):
    """
    Process caption generation
    """
    try:
        # Update job status
        update_job(
            job_id=job_id,
            status="processing",
            progress=10,
            message="Starting caption generation"
        )
        
        # Import caption functions
        from app.core.services.captions import generate_srt_captions, generate_vtt_captions
        
        result = {
            "formats": {}
        }
        
        # Generate SRT if requested
        if "srt" in formats:
            update_job(
                job_id=job_id,
                progress=40,
                message="Generating SRT captions"
            )
            
            srt_content, srt_time = generate_srt_captions(
                chunks,
                include_original,
                include_translation
            )
            
            # Save SRT file
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
                progress=70,
                message="Generating VTT captions"
            )
            
            vtt_content, vtt_time = generate_vtt_captions(
                chunks,
                include_original,
                include_translation
            )
            
            # Save VTT file
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
        
        return {
            "job_id": job_id,
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        # Update job status on error
        update_job(
            job_id=job_id,
            status="failed",
            message=f"Caption generation failed: {str(e)}"
        )
        
        # Re-raise for Celery to handle
        logger.error(f"Caption generation task failed: {str(e)}")
        raise

@app.task(bind=True, name="process_full_pipeline")
def process_full_pipeline(
    self,
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
        
        # Call transcription task synchronously within this task
        transcription_result = process_transcription.s(
            job_id=job_id,
            file_path=file_path,
            language=source_language,
            model_type=transcription_model,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            use_mfa=use_mfa
        ).apply().get()
        
        # Extract transcription result
        if transcription_result and "result" in transcription_result:
            transcription = transcription_result["result"]
            pipeline_results["transcription"] = transcription
            
            # Get chunks for further processing
            transcription_chunks = transcription["chunks"]
            full_text = transcription["full_text"]
            duration = transcription["duration"]
            
            update_job(
                job_id=job_id,
                progress=40,
                message="Transcription completed successfully"
            )
        else:
            # Transcription failed
            raise ValueError("Transcription failed")
        
        # Step 2: Translation (if requested)
        translation_chunks = None
        translation_time = 0
        
        if translate and target_language:
            update_job(
                job_id=job_id,
                progress=50,
                message=f"Starting translation to {target_language}"
            )
            
            # Call translation task synchronously within this task
            translation_result = process_translation.s(
                job_id=job_id,
                chunks=transcription_chunks,
                source_language=source_language,
                target_language=target_language
            ).apply().get()
            
            # Extract translation result
            if translation_result and "result" in translation_result:
                translation = translation_result["result"]
                pipeline_results["translation"] = translation
                
                # Get translated chunks for caption generation
                translation_chunks = translation["chunks"]
                
                update_job(
                    job_id=job_id,
                    progress=70,
                    message="Translation completed successfully"
                )
            else:
                # Translation failed
                update_job(
                    job_id=job_id,
                    message="Translation failed, continuing with transcription only"
                )
        
        # Step 3: Caption generation (if requested)
        if generate_captions:
            update_job(
                job_id=job_id,
                progress=80,
                message="Generating captions"
            )
            
            # Use translation chunks if available, otherwise use transcription chunks
            chunks_for_captions = translation_chunks if translation_chunks else transcription_chunks
            
            # Call caption generation task synchronously within this task
            caption_result = process_captions.s(
                job_id=job_id,
                chunks=chunks_for_captions,
                include_original=include_original,
                include_translation=include_translation and translate,
                formats=caption_formats
            ).apply().get()
            
            # Extract caption result
            if caption_result and "result" in caption_result:
                captions = caption_result["result"]
                pipeline_results["captions"] = captions
                
                update_job(
                    job_id=job_id,
                    progress=90,
                    message="Caption generation completed successfully"
                )
            else:
                # Caption generation failed
                update_job(
                    job_id=job_id,
                    message="Caption generation failed, continuing with transcription/translation only"
                )
        
        # Calculate total processing time
        total_processing_time = time.time() - start_time
        
        # Add processing stats
        pipeline_results["processing_stats"] = {
            "transcription_time": transcription.get("processing_time", 0),
            "translation_time": translation.get("processing_time", 0) if "translation" in pipeline_results else None,
            "caption_generation_time": 0,  # Not tracked separately yet
            "total_processing_time": total_processing_time
        }
        
        # Update job with complete results
        update_job(
            job_id=job_id,
            status="completed",
            progress=100,
            message="Pipeline processing completed successfully",
            result={"pipeline_results": pipeline_results}
        )
        
        return {
            "job_id": job_id,
            "status": "completed",
            "pipeline_results": pipeline_results
        }
    
    except Exception as e:
        # Update job status on error
        update_job(
            job_id=job_id,
            status="failed",
            message=f"Pipeline processing failed: {str(e)}"
        )
        
        # Re-raise for Celery to handle
        logger.error(f"Pipeline task failed: {str(e)}")
        raise
    
    finally:
        # Clean up temporary file
        clean_temp_file(file_path)