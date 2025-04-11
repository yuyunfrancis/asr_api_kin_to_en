from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum

class LanguageEnum(str, Enum):
    english = "english"
    kinyarwanda = "kinyarwanda"

class ModelTypeEnum(str, Enum):
    whisper = "whisper"
    nemo = "nemo"
    whisper_kinyarwanda = "whisper-kinyarwanda"

class TranscriptionRequest(BaseModel):
    language: LanguageEnum = Field(..., description="Language of the audio")
    model_type: ModelTypeEnum = Field(ModelTypeEnum.whisper, description="Model to use for transcription")
    chunk_size: Optional[int] = Field(10, description="Size of audio chunks in seconds")
    overlap_size: Optional[int] = Field(5, description="Overlap between chunks in seconds")
    use_mfa: Optional[bool] = Field(False, description="Use Montreal Forced Aligner (if available)")

class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source_language: LanguageEnum = Field(..., description="Source language")
    target_language: LanguageEnum = Field(..., description="Target language")

class TranslationOptions(BaseModel):
    enabled: bool = Field(False, description="Enable translation")
    target_language: Optional[LanguageEnum] = Field(None, description="Target language for translation")

class CaptionOptions(BaseModel):
    enabled: bool = Field(False, description="Enable caption generation")
    formats: List[Literal["srt", "vtt"]] = Field(["srt", "vtt"], description="Caption formats to generate")
    include_original: bool = Field(True, description="Include original text in captions")
    include_translation: bool = Field(True, description="Include translated text in captions")

class ProcessAudioRequest(BaseModel):
    source_language: LanguageEnum = Field(..., description="Language of the audio")
    transcription_model: ModelTypeEnum = Field(ModelTypeEnum.whisper, description="Model to use for transcription")
    transcription_options: Optional[Dict[str, Any]] = Field({
        "chunk_size": 10,
        "overlap_size": 5,
        "use_mfa": False
    }, description="Transcription-specific options")
    translation: Optional[TranslationOptions] = Field(TranslationOptions(), description="Translation options")
    captions: Optional[CaptionOptions] = Field(CaptionOptions(), description="Caption generation options")
    processing_mode: Literal["sync", "async"] = Field("async", description="Processing mode")