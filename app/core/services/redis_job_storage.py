import uuid
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import redis

from app import config

# Configure logging
logger = logging.getLogger("api.services.jobs")

# Configure Redis connection
redis_client = redis.Redis(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    db=config.REDIS_DB,
    decode_responses=True
)

# Redis keys
JOB_KEY_PREFIX = "job:"
JOB_INDEX_KEY = "jobs:index"
JOB_TYPE_INDEX_PREFIX = "jobs:type:"

def create_job(job_type: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a new job and store it in Redis
    
    Args:
        job_type: Type of job
        metadata: Additional job metadata
        
    Returns:
        Job object
    """
    job_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
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
    
    # Store job in Redis
    job_key = f"{JOB_KEY_PREFIX}{job_id}"
    redis_client.set(job_key, json.dumps(job))
    
    # Add to job index (for listing jobs)
    redis_client.zadd(JOB_INDEX_KEY, {job_id: time.time()})
    
    # Add to job type index
    type_index_key = f"{JOB_TYPE_INDEX_PREFIX}{job_type}"
    redis_client.zadd(type_index_key, {job_id: time.time()})
    
    logger.info(f"Created job {job_id} of type {job_type}")
    
    return job

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get job by ID from Redis
    
    Args:
        job_id: Job ID
        
    Returns:
        Job object if found, None otherwise
    """
    job_key = f"{JOB_KEY_PREFIX}{job_id}"
    job_data = redis_client.get(job_key)
    
    if not job_data:
        return None
        
    try:
        return json.loads(job_data)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode job data for {job_id}")
        return None

def update_job(
    job_id: str, 
    status: Optional[str] = None, 
    progress: Optional[float] = None, 
    message: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Update job status in Redis
    
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
    
    # Update job fields
    if status:
        job["status"] = status
        
    if progress is not None:
        job["progress"] = progress
        
    if message:
        job["message"] = message
        
    if result is not None:
        job["result"] = result
        
    job["updated_at"] = datetime.now().isoformat()
    
    # Save updated job
    job_key = f"{JOB_KEY_PREFIX}{job_id}"
    redis_client.set(job_key, json.dumps(job))
    
    logger.info(f"Updated job {job_id}: status={status}, progress={progress}")
    
    # If job is completed or failed, update its score in the index for sorting
    if status in ["completed", "failed"]:
        redis_client.zadd(JOB_INDEX_KEY, {job_id: time.time()})
        type_index_key = f"{JOB_TYPE_INDEX_PREFIX}{job['job_type']}"
        redis_client.zadd(type_index_key, {job_id: time.time()})
    
    return job

def list_jobs(limit: int = 100, skip: int = 0, job_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List jobs from Redis with pagination
    
    Args:
        limit: Maximum number of jobs to return
        skip: Number of jobs to skip
        job_type: Filter by job type
        
    Returns:
        List of job objects
    """
    # Determine which index to use
    if job_type:
        index_key = f"{JOB_TYPE_INDEX_PREFIX}{job_type}"
    else:
        index_key = JOB_INDEX_KEY
    
    # Get job IDs from sorted set (newest first)
    job_ids = redis_client.zrevrange(index_key, skip, skip + limit - 1)
    
    # Get job data for each ID
    jobs = []
    for job_id in job_ids:
        job = get_job(job_id)
        if job:
            jobs.append(job)
    
    return jobs

def clean_old_jobs(max_age_hours: int = 24) -> int:
    """
    Remove old completed jobs from Redis
    
    Args:
        max_age_hours: Maximum age of jobs in hours
        
    Returns:
        Number of jobs removed
    """
    # Calculate cutoff time
    cutoff_time = time.time() - (max_age_hours * 3600)
    
    # Get all job IDs
    job_ids = redis_client.zrange(JOB_INDEX_KEY, 0, -1, withscores=True)
    
    count = 0
    for job_id, score in job_ids:
        # Skip recent jobs
        if score > cutoff_time:
            continue
            
        # Check if job is completed or failed
        job = get_job(job_id)
        if job and job["status"] in ["completed", "failed"]:
            # Remove from Redis
            job_key = f"{JOB_KEY_PREFIX}{job_id}"
            redis_client.delete(job_key)
            
            # Remove from indexes
            redis_client.zrem(JOB_INDEX_KEY, job_id)
            type_index_key = f"{JOB_TYPE_INDEX_PREFIX}{job['job_type']}"
            redis_client.zrem(type_index_key, job_id)
            
            count += 1
            
    logger.info(f"Cleaned {count} old jobs")
    
    return count

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
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }
    
    # Include result if available
    if job["status"] == "completed" and job["result"]:
        serialized["result"] = job["result"]
        
    return serialized