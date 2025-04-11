import uuid
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import json

# Configure logging
logger = logging.getLogger("api.services.jobs")

# Simple in-memory storage for jobs
# In a production system, this would be replaced with a database
JOB_STORAGE = {}

def create_job(job_type: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a new job and return the job object
    
    Args:
        job_type: Type of job
        metadata: Additional job metadata
        
    Returns:
        Job object
    """
    job_id = str(uuid.uuid4())
    created_at = datetime.now()
    
    job = {
        "job_id": job_id,
        "job_type": job_type,
        "status": "pending",
        "progress": 0.0,
        "message": "Job created",
        "created_at": created_at,
        "updated_at": created_at,
        "metadata": metadata or {},
        "result": None
    }
    
    JOB_STORAGE[job_id] = job
    logger.info(f"Created job {job_id} of type {job_type}")
    
    return job

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get job by ID
    
    Args:
        job_id: Job ID
        
    Returns:
        Job object if found, None otherwise
    """
    return JOB_STORAGE.get(job_id)

def update_job(
    job_id: str, 
    status: Optional[str] = None, 
    progress: Optional[float] = None, 
    message: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Update job status
    
    Args:
        job_id: Job ID
        status: New status
        progress: Progress percentage (0-100)
        message: Status message
        result: Job result data
        
    Returns:
        Updated job object if found, None otherwise
    """
    job = get_job(job_id)
    
    if not job:
        logger.warning(f"Attempted to update non-existent job {job_id}")
        return None
    
    if status:
        job["status"] = status
        
    if progress is not None:
        job["progress"] = progress
        
    if message:
        job["message"] = message
        
    if result is not None:
        job["result"] = result
        
    job["updated_at"] = datetime.now()
    
    logger.info(f"Updated job {job_id}: status={status}, progress={progress}")
    
    return job

def list_jobs(limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
    """
    List jobs with pagination
    
    Args:
        limit: Maximum number of jobs to return
        skip: Number of jobs to skip
        
    Returns:
        List of job objects
    """
    jobs = list(JOB_STORAGE.values())
    # Sort by creation time, newest first
    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    
    return jobs[skip:skip+limit]

def clean_old_jobs(max_age_hours: int = 24) -> int:
    """
    Remove old completed jobs from storage
    
    Args:
        max_age_hours: Maximum age of jobs in hours
        
    Returns:
        Number of jobs removed
    """
    now = datetime.now()
    job_ids_to_remove = []
    
    for job_id, job in JOB_STORAGE.items():
        age = now - job["created_at"]
        
        # Convert to hours
        age_hours = age.total_seconds() / 3600
        
        if (age_hours > max_age_hours and 
            job["status"] in ["completed", "failed"]):
            job_ids_to_remove.append(job_id)
            
    # Remove jobs
    for job_id in job_ids_to_remove:
        del JOB_STORAGE[job_id]
        
    logger.info(f"Cleaned {len(job_ids_to_remove)} old jobs")
    
    return len(job_ids_to_remove)

def serialize_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize job object for API response
    
    Args:
        job: Job object
        
    Returns:
        Serialized job object
    """
    serialized = {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "created_at": job["created_at"].isoformat(),
        "updated_at": job["updated_at"].isoformat(),
    }
    
    # Include result if available
    if job["status"] == "completed" and job["result"]:
        serialized["result"] = job["result"]
        
    return serialized