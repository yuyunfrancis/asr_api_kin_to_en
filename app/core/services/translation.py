import logging
import re
import torch
import time
from typing import List, Dict, Any, Tuple, Optional

from app import config
from app.core.models.translation import (
    load_translation_model,
    get_language_direction,
    get_language_codes,
    detect_model_type
)

# Configure logging
logger = logging.getLogger("api.services.translation")

async def translate_text(
    text: str, 
    source_language: str, 
    target_language: str, 
    batch_size: int = 512
) -> Tuple[str, float]:
    """
    Translate text between languages
    
    Args:
        text: Text to translate
        source_language: Source language
        target_language: Target language
        batch_size: Maximum number of characters to process at once
        
    Returns:
        Tuple containing translated text and processing time
    """
    start_time = time.time()
    
    # Determine translation direction
    direction = get_language_direction(source_language, target_language)
    logger.info(f"Translating text of length {len(text)} from {source_language} to {target_language}")
    
    # Load models
    tokenizer, model = load_translation_model(direction)
    
    # Determine model type
    model_type = detect_model_type(model.config._name_or_path)
    
    # Get specific language codes for the model
    source_lang_code, target_lang_code = get_language_codes(
        target_language if direction == "en-to-kin" else source_language, 
        model_type
    )
    
    # Check if using NLLB model which requires specific language codes
    is_nllb_model = model_type == "nllb"
    
    # If text is short enough, translate it all at once
    if len(text) < batch_size:
        if is_nllb_model:
            # NLLB model uses forced_bos_token_id directly
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            # Get the language token ID - NLLB uses different methods than other models
            if hasattr(tokenizer, 'lang_code_to_id'):
                # For older versions of the tokenizer
                forced_bos_token_id = tokenizer.lang_code_to_id[target_lang_code]
            else:
                # For newer versions, use the convert_tokens_to_ids method
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang_code)
            
            # Move to GPU if available
            if config.gpu_available:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                model.to("cuda")
            
            # Generate translation with forced BOS token
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, 
                    forced_bos_token_id=forced_bos_token_id
                )
        else:
            # Regular models like MarianMT
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            
            # Move to GPU if available
            if config.gpu_available:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                model.to("cuda")
            
            # Generate translation
            with torch.no_grad():
                output_ids = model.generate(**inputs)
        
        # Decode the generated tokens
        translation = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return translation, time.time() - start_time
    
    # For longer text, split into sentences and translate in batches
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Group sentences into batches
    batches = []
    current_batch = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) > batch_size and current_batch:
            batches.append(" ".join(current_batch))
            current_batch = [sentence]
            current_length = len(sentence)
        else:
            current_batch.append(sentence)
            current_length += len(sentence)
    
    if current_batch:
        batches.append(" ".join(current_batch))
    
    # Translate each batch
    translations = []
    for batch in batches:
        if is_nllb_model:
            # NLLB model uses forced_bos_token_id directly
            inputs = tokenizer(batch, return_tensors="pt", padding=True)
            # Get the language token ID - NLLB uses different methods than other models
            if hasattr(tokenizer, 'lang_code_to_id'):
                # For older versions of the tokenizer
                forced_bos_token_id = tokenizer.lang_code_to_id[target_lang_code]
            else:
                # For newer versions, use the convert_tokens_to_ids method
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang_code)
            
            # Move to GPU if available
            if config.gpu_available:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate translation with forced BOS token
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, 
                    forced_bos_token_id=forced_bos_token_id
                )
        else:
            # Regular models like MarianMT
            inputs = tokenizer(batch, return_tensors="pt", padding=True)
            
            # Move to GPU if available
            if config.gpu_available:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                output_ids = model.generate(**inputs)
        
        # Decode the generated tokens
        batch_translation = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        translations.append(batch_translation)
    
    # Join all translations
    complete_translation = " ".join(translations)
    
    return complete_translation, time.time() - start_time

async def translate_chunks(
    chunks: List[Dict[str, Any]], 
    source_language: str, 
    target_language: str
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Translate each chunk separately while preserving timing information
    
    Args:
        chunks: List of dictionaries with start_time, end_time, and text
        source_language: Source language
        target_language: Target language
        
    Returns:
        Tuple of list of dictionaries with start_time, end_time, original_text, 
        and translated_text, and processing time
    """
    start_time = time.time()
    logger.info(f"Translating {len(chunks)} chunks from {source_language} to {target_language}")
    
    # Determine translation direction
    direction = get_language_direction(source_language, target_language)
    
    # Load models
    tokenizer, model = load_translation_model(direction)
    
    # Determine model type
    model_type = detect_model_type(model.config._name_or_path)
    
    # Get specific language codes for the model
    source_lang_code, target_lang_code = get_language_codes(
        target_language if direction == "en-to-kin" else source_language, 
        model_type
    )
    
    # Check if using NLLB model which requires specific language codes
    is_nllb_model = model_type == "nllb"
    
    translated_chunks = []
    
    for chunk in chunks:
        # Skip empty chunks
        if not chunk.get("text", "").strip():
            continue
            
        # Prepare inputs based on model type
        if is_nllb_model:
            # NLLB model uses forced_bos_token_id directly
            inputs = tokenizer(chunk["text"], return_tensors="pt", padding=True)
            # Get the language token ID - NLLB uses different methods than other models
            if hasattr(tokenizer, 'lang_code_to_id'):
                # For older versions of the tokenizer
                forced_bos_token_id = tokenizer.lang_code_to_id[target_lang_code]
            else:
                # For newer versions, use the convert_tokens_to_ids method
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang_code)
            
            # Move to GPU if available
            if config.gpu_available:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate translation with forced BOS token
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, 
                    forced_bos_token_id=forced_bos_token_id
                )
        else:
            # Regular models like MarianMT
            inputs = tokenizer(chunk["text"], return_tensors="pt", padding=True)
            
            # Move to GPU if available
            if config.gpu_available:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                output_ids = model.generate(**inputs)
        
        # Decode the generated tokens
        translation = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        translated_chunks.append({
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "original_text": chunk["text"],
            "translated_text": translation
        })
    
    processing_time = time.time() - start_time
    return translated_chunks, processing_time