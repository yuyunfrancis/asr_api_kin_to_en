import os
import uuid
import shutil
from pathlib import Path
from fastapi import UploadFile
from datetime import datetime, timedelta
import logging

from app import config

# Configure logging
logger = logging.getLogger("api")

def generate_unique_filename(original_filename: str) -> str:
    """Generate a unique filename while preserving extension"""
    name, ext = os.path.splitext(original_filename)
    unique_id = str(uuid.uuid4())
    return f"{unique_id}{ext}"

async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save uploaded file to temporary directory
    
    Args:
        upload_file: The uploaded file
        
    Returns:
        str: Path to the saved file
    """
    # Ensure temp directory exists
    os.makedirs(config.TEMP_FILE_DIR, exist_ok=True)
    
    # Generate unique filename
    unique_filename = generate_unique_filename(upload_file.filename)
    file_path = os.path.join(config.TEMP_FILE_DIR, unique_filename)
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        logger.info(f"Saved uploaded file to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise e
    finally:
        upload_file.file.close()
        
def clean_temp_file(file_path: str) -> bool:
    """
    Remove temporary file
    
    Args:
        file_path: Path to the file to remove
        
    Returns:
        bool: True if file was removed successfully
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed temporary file: {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error removing temporary file: {e}")
        return False

def save_caption_file(content: str, job_id: str, format_type: str = "srt") -> str:
    """
    Save caption content to a file
    
    Args:
        content: Caption content
        job_id: Job ID to use in filename
        format_type: Caption format (srt or vtt)
        
    Returns:
        str: Path to the saved file
    """
    # Create directory if it doesn't exist
    caption_dir = os.path.join(config.TEMP_FILE_DIR, "captions", job_id)
    os.makedirs(caption_dir, exist_ok=True)
    
    # Save file
    file_path = os.path.join(caption_dir, f"captions.{format_type}")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved caption file to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving caption file: {e}")
        raise e

def get_caption_file_url(job_id: str, format_type: str = "srt") -> str:
    """
    Get URL for caption file
    
    Args:
        job_id: Job ID
        format_type: Caption format (srt or vtt)
        
    Returns:
        str: URL for the caption file
    """
    return f"/api/files/captions/{job_id}/{format_type}"

def cleanup_old_files(max_age_hours: int = 24) -> int:
    """
    Clean up old temporary files
    
    Args:
        max_age_hours: Maximum age of files in hours
        
    Returns:
        int: Number of files removed
    """
    count = 0
    max_age = datetime.now() - timedelta(hours=max_age_hours)
    
    for root, _, files in os.walk(config.TEMP_FILE_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_modified < max_age:
                    os.remove(file_path)
                    count += 1
            except Exception as e:
                logger.error(f"Error removing old file {file_path}: {e}")
    
    return count