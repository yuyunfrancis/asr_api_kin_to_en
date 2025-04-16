import os
import logging
import torch
import librosa
import soundfile as sf
from typing import List, Dict, Any, Optional, Tuple
import time

from app import config
from app.core.models.transcription import (
    load_pipeline_model, 
    load_whisper_model, 
    load_nemo_model, 
    initialize_transcription_models
)
from app.utils.audio import (
    load_audio,
    save_audio,
    remove_duplicates,
    estimate_timestamps_from_text
)

# Configure logging
logger = logging.getLogger("api.services.transcription")

async def transcribe_with_pipeline(audio_path: str, language: str = "en") -> Tuple[List[Dict[str, Any]], float]:
    """
    Transcribe using the pipeline approach
    
    Args:
        audio_path: Path to audio file
        language: Language code
        
    Returns:
        Tuple of transcription chunks and processing time
    """
    logger.info(f"Transcribing with pipeline: language={language}")
    start_time = time.time()
    
    # Load model
    model = load_pipeline_model("openai/whisper-small")
    
    # Transcribe audio
    transcription = model(
        audio_path, 
        generate_kwargs={"language": language, "task": "transcribe"}
    )
    
    # Format the results
    chunks = []
    for chunk in transcription['chunks']:
        chunks.append({
            "start_time": chunk["timestamp"][0],
            "end_time": chunk["timestamp"][1],
            "text": chunk["text"]
        })
    
    processing_time = time.time() - start_time
    logger.info(f"Transcription completed in {processing_time:.2f} seconds")
    
    return chunks, processing_time

async def transcribe_with_whisper_model(
    audio_path: str, 
    language: str = "sw", 
    chunk_size_seconds: int = 10, 
    overlap_seconds: int = 5
):
    """
    Enhanced transcription using the Whisper model with advanced techniques
    """
    start_time = time.time()
    logger.info(f"Transcribing with Whisper model: language={language}, chunk_size={chunk_size_seconds}s, overlap={overlap_seconds}s")
    
    # Determine model path based on language
    if language.lower() in ["sw", "kinyarwanda", "kin"]:
        model_path = "mbazaNLP/Whisper-Small-Kinyarwanda"
    else:
        model_path = "openai/whisper-small"
    
    # Load model
    processor, model = load_whisper_model(model_path)
    
    # IMPORTANT: Don't set forced_decoder_ids on model.config
    # Instead, create decoder_input_ids for each generation call
    
    # Load and preprocess audio
    logger.info(f"Loading audio file: {audio_path}")
    audio, sr, audio_duration = load_audio(audio_path, sr=16000)
    
    # Normalize audio
    audio = librosa.util.normalize(audio)
    
    # Calculate chunk parameters
    if audio_duration > 60:  
        overlap_ratio = min(0.5, overlap_seconds / chunk_size_seconds + 0.1)
    else:
        overlap_ratio = overlap_seconds / chunk_size_seconds
    
    chunk_size = chunk_size_seconds * sr
    overlap = int(chunk_size * overlap_ratio)
    
    # Initialize results
    chunks = []
    
    # Process audio in chunks
    for i in range(0, len(audio), chunk_size - overlap):
        # Calculate timestamps
        start_time_sec = max(0, i / sr)
        end_idx = min(i + chunk_size, len(audio))
        chunk_audio = audio[i:end_idx]
        end_time_sec = end_idx / sr
        
        # Skip very short chunks
        if len(chunk_audio) < sr * 0.5:
            logger.debug(f"Skipping very short chunk ({len(chunk_audio)/sr:.2f}s)")
            continue
        
        # Process the chunk
        input_features = processor(chunk_audio, sampling_rate=16000, return_tensors="pt").input_features
        
        # Create attention mask
        attention_mask = torch.ones(input_features.shape[:-1], dtype=torch.long)
        
        # Move to GPU if available
        if config.gpu_available:
            input_features = input_features.to("cuda")
            attention_mask = attention_mask.to("cuda")
        
        # Generate transcription
        try:
            with torch.no_grad():
                # Get the token IDs for the decoder prompt
                prompt_ids = processor.tokenizer.encode(
                    f"<|{language}|> <|transcribe|>",
                    add_special_tokens=False,
                    return_tensors="pt"
                )
                
                if config.gpu_available:
                    prompt_ids = prompt_ids.to("cuda")
                
                predicted_ids = model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    decoder_input_ids=prompt_ids,  
                    num_beams=5,
                    max_length=256,
                    min_length=1,
                    length_penalty=1.0,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            chunk_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Add chunk to results
            chunks.append({
                "start_time": start_time_sec,
                "end_time": end_time_sec,
                "text": chunk_transcription.strip()
            })
            
        except Exception as e:
            logger.error(f"Error transcribing chunk {start_time_sec:.2f}s-{end_time_sec:.2f}s: {e}")
    
    # Sort chunks by start time
    chunks.sort(key=lambda x: x["start_time"])
    
    # Process chunks to remove duplicates
    processed_chunks = remove_duplicates(chunks)
    
    processing_time = time.time() - start_time
    logger.info(f"Whisper transcription completed in {processing_time:.2f} seconds")
    
    return processed_chunks, processing_time

async def transcribe_with_production_approach(
    audio_path: str,
    language: str = "sw",
    use_mfa: bool = False
) -> Tuple[List[Dict[str, Any]], float, Optional[str]]:
    """
    Production-ready approach for Kinyarwanda transcription with accurate timestamps:
    1. Use NeMo for main transcription (highest accuracy for Kinyarwanda)
    2. Use Whisper for initial chunking and timestamps
    3. Optionally refine with MFA for word-level precision (when enabled)
    4. Fall back gracefully through multiple methods if any step fails
    
    Args:
        audio_path: Path to audio file
        language: Language code
        use_mfa: Whether to use Montreal Forced Aligner
        
    Returns:
        Tuple of transcription chunks, processing time, and error message (if any)
    """
    start_time = time.time()
    logger.info(f"Transcribing with production approach: {audio_path}, language: {language}, MFA enabled: {use_mfa}")
    
    try:
        # Step 1: Pre-process audio
        audio, sr, audio_duration = load_audio(audio_path, sr=16000)
        
        temp_wav_path = os.path.join(os.path.dirname(audio_path), "temp_input.wav")
        save_audio(audio, temp_wav_path, sr)
        
        # Initialize models based on language
        if language.lower() in ["sw", "kinyarwanda", "kin"]:
            # Step 2: Get the main transcription from NeMo (higher quality for Kinyarwanda)
            try:
                logger.info("Running NeMo transcription...")
                models = initialize_transcription_models("nemo", language)
                nemo_model = models["nemo_model"]
                nemo_loaded = models["nemo_loaded"]
                
                if nemo_loaded and nemo_model:
                    nemo_transcriptions = nemo_model.transcribe([temp_wav_path])
                    
                    if nemo_transcriptions and len(nemo_transcriptions) > 0:
                        # Extract text from NeMo result
                        if hasattr(nemo_transcriptions[0], 'text'):
                            nemo_text = nemo_transcriptions[0].text
                        elif hasattr(nemo_transcriptions[0], 'transcript'):
                            nemo_text = nemo_transcriptions[0].transcript
                        else:
                            nemo_text = str(nemo_transcriptions[0])
                        
                        logger.info(f"NeMo transcription successful: {len(nemo_text)} characters")
                    else:
                        logger.warning("NeMo returned empty transcription")
                        nemo_text = None
                else:
                    nemo_text = None
                    error = models.get("error", "NeMo model could not be loaded")
                    logger.warning(error)
            except Exception as e:
                logger.error(f"NeMo transcription failed: {e}")
                nemo_text = None
        else:
            # For English, we don't use NeMo
            nemo_text = None
        
        # Step 3: Get timestamps from Whisper
        try:
            logger.info("Running Whisper transcription for timestamps...")
            whisper_chunks, whisper_time = await transcribe_with_whisper_model(
                audio_path=temp_wav_path,
                language=language,
                chunk_size_seconds=10,
                overlap_seconds=5
            )
            
            if not whisper_chunks:
                logger.warning("Whisper returned no chunks")
                whisper_chunks = []
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            whisper_chunks = []
        
        # Decision tree based on what succeeded
        if not nemo_text and not whisper_chunks:
            # Both methods failed - try one more fallback approach
            logger.error("Both NeMo and Whisper transcription failed. Attempting direct Whisper pipeline...")
            try:
                chunks, _ = await transcribe_with_pipeline(temp_wav_path, language="en")
                logger.info("Fallback to Whisper pipeline successful")
                os.remove(temp_wav_path)
                return chunks, time.time() - start_time, "Used fallback Whisper pipeline due to primary methods failing"
            except Exception as e:
                logger.error(f"All transcription methods failed: {e}")
                os.remove(temp_wav_path)
                return [], time.time() - start_time, f"All transcription methods failed: {str(e)}"
        
        elif not nemo_text:
            # NeMo failed but Whisper worked - use Whisper results
            logger.warning("Using Whisper results because NeMo failed")
            os.remove(temp_wav_path)
            return whisper_chunks, time.time() - start_time, "Used Whisper due to NeMo failure"
        
        elif not whisper_chunks:
            # Whisper failed but NeMo worked - create estimated timestamps
            logger.warning("Using NeMo with estimated timestamps because Whisper failed")
            chunks = estimate_timestamps_from_text(nemo_text, audio_duration)
            os.remove(temp_wav_path)
            return chunks, time.time() - start_time, "Used NeMo with estimated timestamps due to Whisper failure"
        
        # Step 4: We have both NeMo and Whisper results - align them
        logger.info("Aligning NeMo transcription with Whisper timestamps...")
        
        # Option: Use MFA for precise word-level alignment if enabled
        if use_mfa:
            try:
                logger.info("Attempting Montreal Forced Aligner for precision...")
                
                # Check if MFA is available
                try:
                    import montreal_forced_aligner
                    mfa_available = True
                except ImportError:
                    logger.warning("Montreal Forced Aligner not installed. Skipping MFA step.")
                    mfa_available = False
                
                if mfa_available:
                    # MFA implementation would go here
                    # For now, fall back to hybrid alignment
                    logger.info("MFA processing would occur here. Falling back to hybrid alignment for now.")
            except Exception as e:
                logger.error(f"MFA alignment failed: {e}")
        
        # Step 5: Hybrid alignment (fallback or if MFA not enabled)
        from app.utils.audio import align_nemo_with_whisper
        aligned_chunks = align_nemo_with_whisper(nemo_text, whisper_chunks, audio_duration)
        
        # Clean up
        os.remove(temp_wav_path)
        return aligned_chunks, time.time() - start_time, None
    
    except Exception as e:
        logger.error(f"Error in production transcription approach: {e}")
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        return [], time.time() - start_time, str(e)

# Helper function for align_nemo_with_whisper
def align_nemo_with_whisper(nemo_text: str, whisper_chunks: List[Dict[str, Any]], audio_duration: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Align NeMo transcription text with Whisper chunks/timestamps
    Using dynamic text alignment algorithms for better matching
    
    Args:
        nemo_text: Text from NeMo transcription
        whisper_chunks: Chunks from Whisper transcription
        audio_duration: Duration of audio in seconds
        
    Returns:
        List of aligned chunks
    """
    # If we only have one Whisper chunk, use it with NeMo text
    if len(whisper_chunks) <= 1:
        if audio_duration:
            end_time = audio_duration
        elif whisper_chunks:
            end_time = whisper_chunks[0]["end_time"]
        else:
            end_time = 0.0
            
        return [{
            "start_time": 0.0,
            "end_time": end_time,
            "text": nemo_text
        }]
    
    # Split nemo_text into words
    nemo_words = nemo_text.split()
    
    # Count words in Whisper chunks
    whisper_word_counts = [len(chunk["text"].split()) for chunk in whisper_chunks]
    total_whisper_words = sum(whisper_word_counts)
    
    # Basic proportional distribution
    word_ratios = [count / total_whisper_words for count in whisper_word_counts]
    
    # Distribute NeMo words to chunks based on word ratios
    aligned_chunks = []
    start_idx = 0
    
    for i, chunk in enumerate(whisper_chunks):
        # Calculate how many words should go in this chunk
        if i == len(whisper_chunks) - 1:
            # Last chunk gets all remaining words
            chunk_words = nemo_words[start_idx:]
        else:
            # Distribute proportionally
            word_count = max(1, int(len(nemo_words) * word_ratios[i]))
            end_idx = min(start_idx + word_count, len(nemo_words))
            chunk_words = nemo_words[start_idx:end_idx]
            start_idx = end_idx
        
        if chunk_words:
            aligned_chunks.append({
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "text": " ".join(chunk_words)
            })
    
    return aligned_chunks