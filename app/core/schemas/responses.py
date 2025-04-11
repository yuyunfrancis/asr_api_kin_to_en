from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"

class TranscriptionChunk(BaseModel):
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text")

class TranslationChunk(BaseModel):
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    original_text: str = Field(..., description="Original text")
    translated_text: str = Field(..., description="Translated text")

class TranscriptionResult(BaseModel):
    language: str = Field(..., description="Language of the transcription")
    model_used: str = Field(..., description="Model used for transcription")
    full_text: str = Field(..., description="Complete transcribed text")
    chunks: List[TranscriptionChunk] = Field(..., description="Timestamped chunks")
    confidence_score: Optional[float] = Field(None, description="Confidence score (if available)")
    duration: float = Field(..., description="Audio duration in seconds")

class TranslationResult(BaseModel):
    source_language: str = Field(..., description="Source language")
    target_language: str = Field(..., description="Target language")
    model_used: str = Field(..., description="Model used for translation")
    full_text: str = Field(..., description="Complete translated text")
    chunks: Optional[List[TranslationChunk]] = Field(None, description="Timestamped chunks (if available)")

class CaptionResult(BaseModel):
    formats: Dict[str, Dict[str, str]] = Field(..., description="Caption files by format")
    metadata: Dict[str, Any] = Field(..., description="Caption metadata")

class ProcessingStats(BaseModel):
    transcription_time: Optional[float] = Field(None, description="Transcription processing time in seconds")
    translation_time: Optional[float] = Field(None, description="Translation processing time in seconds")
    caption_generation_time: Optional[float] = Field(None, description="Caption generation time in seconds")
    total_processing_time: float = Field(..., description="Total processing time in seconds")

class JobStatusResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    message: Optional[str] = Field(None, description="Status message or error details")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last status update timestamp")
    result_url: Optional[str] = Field(None, description="URL to access results when completed")

class PipelineResults(BaseModel):
    transcription: Optional[TranscriptionResult] = Field(None, description="Transcription results")
    translation: Optional[TranslationResult] = Field(None, description="Translation results")
    captions: Optional[CaptionResult] = Field(None, description="Caption results")
    processing_stats: Optional[ProcessingStats] = Field(None, description="Processing statistics")

class PipelineResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    pipeline_results: Optional[PipelineResults] = Field(None, description="Results of the pipeline processing")

class ModelsResponse(BaseModel):
    transcription_models: List[Dict[str, Any]] = Field(..., description="Available transcription models")
    translation_models: List[Dict[str, Any]] = Field(..., description="Available translation models")