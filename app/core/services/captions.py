import logging
from datetime import timedelta
import time
from typing import List, Dict, Any, Tuple

from app.utils.file_utils import save_caption_file

# Configure logging
logger = logging.getLogger("api.services.captions")

def format_timestamp(seconds: float, format_type: str = "srt") -> str:
    """
    Convert seconds to formatted timestamp
    
    Args:
        seconds: Time in seconds
        format_type: Format type (srt or vtt)
        
    Returns:
        Formatted timestamp string
    """
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    
    if format_type == "srt":
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    elif format_type == "vtt":
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

async def generate_srt_captions(
    chunks: List[Dict[str, Any]], 
    include_original: bool = True, 
    include_translation: bool = True
) -> Tuple[str, float]:
    """
    Generate SRT format captions
    
    Args:
        chunks: List of chunks with timing and text information
        include_original: Whether to include original text
        include_translation: Whether to include translated text
        
    Returns:
        Tuple of SRT content string and processing time
    """
    
    logger.info(f"Generating SRT captions from {len(chunks)} chunks")
    logger.info(f"First chunk sample: {chunks[0] if chunks else 'No chunks'}")
    start_time = time.time()
    logger.info(f"Generating SRT captions from {len(chunks)} chunks")
    
    srt_content = []
    
    for i, chunk in enumerate(chunks, 1):
        # Skip invalid chunks
        has_original = "original_text" in chunk or "text" in chunk
        has_translation = "translated_text" in chunk
        
        if not has_original and not has_translation:
            continue
            
        # Get text content
        original_text = chunk.get("original_text", chunk.get("text", ""))
        translated_text = chunk.get("translated_text", "")
        
        # Format timestamps
        start_timestamp = format_timestamp(chunk["start_time"], "srt")
        end_timestamp = format_timestamp(chunk["end_time"], "srt")
        
        # Create caption entry
        srt_content.append(f"{i}")
        srt_content.append(f"{start_timestamp} --> {end_timestamp}")
        
        # Add text based on options
        if include_original and has_original and original_text.strip():
            srt_content.append(f"{original_text}")
        
        if include_translation and has_translation and translated_text.strip():
            if include_original and has_original:
                srt_content.append(f"{translated_text}")
            else:
                srt_content.append(f"{translated_text}")
                
        srt_content.append("")  # Empty line between entries
    
    # Join content into single string
    srt_text = "\n".join(srt_content)
    
    processing_time = time.time() - start_time
    logger.info(f"SRT caption generation completed in {processing_time:.2f} seconds")
    
    return srt_text, processing_time

async def generate_vtt_captions(
    chunks: List[Dict[str, Any]], 
    include_original: bool = True, 
    include_translation: bool = True
) -> Tuple[str, float]:
    """
    Generate WebVTT format captions
    
    Args:
        chunks: List of chunks with timing and text information
        include_original: Whether to include original text
        include_translation: Whether to include translated text
        
    Returns:
        Tuple of VTT content string and processing time
    """
    start_time = time.time()
    logger.info(f"Generating VTT captions from {len(chunks)} chunks")
    
    # Start with VTT header
    vtt_content = ["WEBVTT", ""]
    
    for i, chunk in enumerate(chunks):
        # Skip invalid chunks
        has_original = "original_text" in chunk or "text" in chunk
        has_translation = "translated_text" in chunk
        
        if not has_original and not has_translation:
            continue
            
        # Get text content
        original_text = chunk.get("original_text", chunk.get("text", ""))
        translated_text = chunk.get("translated_text", "")
        
        # Format timestamps
        start_timestamp = format_timestamp(chunk["start_time"], "vtt")
        end_timestamp = format_timestamp(chunk["end_time"], "vtt")
        
        # Create caption entry
        vtt_content.append(f"{start_timestamp} --> {end_timestamp}")
        
        # Add text based on options
        if include_original and has_original and original_text.strip():
            vtt_content.append(f"{original_text}")
        
        if include_translation and has_translation and translated_text.strip():
            if include_original and has_original:
                vtt_content.append(f"{translated_text}")
            else:
                vtt_content.append(f"{translated_text}")
                
        vtt_content.append("")  # Empty line between entries
    
    # Join content into single string
    vtt_text = "\n".join(vtt_content)
    
    processing_time = time.time() - start_time
    logger.info(f"VTT caption generation completed in {processing_time:.2f} seconds")
    
    return vtt_text, processing_time

# In captions.py
async def save_caption_files(
    chunks: List[Dict[str, Any]], 
    job_id: str,
    include_original: bool = True, 
    include_translation: bool = True
):
    """Generate and save caption files"""
    start_time = time.time()
    
    # Log input parameters
    logger.info(f"Starting caption generation for job {job_id}")
    logger.info(f"Chunks received: {len(chunks)}")
    logger.info(f"Include original: {include_original}, Include translation: {include_translation}")
    
    # Generate SRT content
    srt_content, srt_time = await generate_srt_captions(
        chunks, include_original, include_translation
    )
    
    # Log content length
    logger.info(f"Generated SRT content length: {len(srt_content)}")
    
    # Generate VTT content
    vtt_content, vtt_time = await generate_vtt_captions(
        chunks, include_original, include_translation
    )
    
    logger.info(f"Generated VTT content length: {len(vtt_content)}")
    
    # Save files
    try:
        srt_path = save_caption_file(srt_content, job_id, "srt")
        logger.info(f"SRT file saved to: {srt_path}")
        
        vtt_path = save_caption_file(vtt_content, job_id, "vtt")
        logger.info(f"VTT file saved to: {vtt_path}")
        
        files = {
            "srt": srt_path,
            "vtt": vtt_path
        }
    except Exception as e:
        logger.error(f"Error saving caption files: {str(e)}")
        raise
    
    processing_time = time.time() - start_time
    logger.info(f"Caption generation completed in {processing_time:.2f} seconds")
    
    return files, processing_time
