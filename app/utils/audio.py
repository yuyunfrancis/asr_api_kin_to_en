import os
import logging
import librosa
import soundfile as sf
import re
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logger = logging.getLogger("api")

def load_audio(audio_path: str, sr: int = 16000) -> Tuple[Any, int, float]:
    """
    Load audio file and return audio data, sample rate, and duration
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate
        
    Returns:
        Tuple containing audio data, sample rate, and duration
    """
    try:
        audio, sr = librosa.load(audio_path, sr=sr, mono=True)
        audio_duration = len(audio) / sr
        logger.info(f"Loaded audio file: {audio_path}, duration: {audio_duration:.2f}s")
        return audio, sr, audio_duration
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}")
        raise e

def save_audio(audio_data: Any, path: str, sample_rate: int = 16000) -> str:
    """
    Save audio data to file
    
    Args:
        audio_data: Audio data
        path: Path to save audio file
        sample_rate: Sample rate
        
    Returns:
        Path to saved file
    """
    try:
        sf.write(path, audio_data, sample_rate)
        logger.info(f"Saved audio to {path}")
        return path
    except Exception as e:
        logger.error(f"Error saving audio file {path}: {e}")
        raise e

def calculate_similarity(str1: str, str2: str) -> float:
    """
    Calculate string similarity for overlap detection
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Similarity score between 0 and 1
    """
    # Simple case: exact match
    if str1 == str2:
        return 1.0
        
    # Calculate word overlap
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())
    
    if not words1 or not words2:
        return 0.0
        
    # Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union

def format_transcription(chunks: List[Dict[str, Any]]) -> str:
    """
    Format transcription results with timestamps
    
    Args:
        chunks: List of transcription chunks
        
    Returns:
        Formatted text with timestamps
    """
    formatted_text = ""
    for chunk in chunks:
        start_time = chunk["start_time"]
        end_time = chunk["end_time"]
        text = chunk.get("text", "") 
        formatted_text += f"[{start_time:.2f}:{end_time:.2f}] {text}\n"
    return formatted_text.strip()

def estimate_timestamps_from_text(text: str, audio_duration: float) -> List[Dict[str, Any]]:
    """
    Estimate timestamps from text when Whisper fails
    
    Args:
        text: Transcribed text
        audio_duration: Duration of audio in seconds
        
    Returns:
        List of chunks with estimated timestamps
    """
    # Split text into sentences
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    sentences = [s.strip() for s in sentence_pattern.split(text) if s.strip()]
    
    if not sentences:
        sentences = [text]
    
    # Simple approach: distribute time evenly
    chunks = []
    chunk_duration = audio_duration / len(sentences)
    
    for i, sentence in enumerate(sentences):
        start_time = i * chunk_duration
        end_time = (i + 1) * chunk_duration
        
        chunks.append({
            "start_time": start_time,
            "end_time": end_time,
            "text": sentence
        })
    
    return chunks

def remove_duplicates(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Helper function to remove duplicated content between chunks
    
    Args:
        chunks: List of transcription chunks
        
    Returns:
        List of chunks with duplicates removed
    """
    if not chunks:
        return []
        
    processed_chunks = [chunks[0]]
    
    for i in range(1, len(chunks)):
        current_chunk = chunks[i]
        
        # Skip empty chunks
        if not current_chunk["text"]:
            continue
            
        # Check for overlap with previous chunk
        prev_chunk = processed_chunks[-1]
        prev_words = prev_chunk["text"].split()
        curr_words = current_chunk["text"].split()
        
        # Try different overlap sizes to find repetition
        repeated_text = False
        for overlap_size in range(min(len(prev_words), len(curr_words), 10), 2, -1):
            if len(prev_words) >= overlap_size and len(curr_words) >= overlap_size:
                prev_phrase = " ".join(prev_words[-overlap_size:]).lower()
                curr_phrase = " ".join(curr_words[:overlap_size]).lower()
                
                # If significant similarity, remove overlapping portion
                if calculate_similarity(prev_phrase, curr_phrase) > 0.7:
                    current_chunk["text"] = " ".join(curr_words[overlap_size:])
                    repeated_text = True
                    break
        
        # If chunk still has content after removing duplicates, add it
        if current_chunk["text"] and not repeated_text:
            processed_chunks.append(current_chunk)
        elif repeated_text and current_chunk["text"]:
            processed_chunks.append(current_chunk)
    
    return processed_chunks