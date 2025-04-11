from fastapi import APIRouter, HTTPException, Query, Path, Depends
from typing import List, Dict, Any, Optional

from app.core.services.job_storage import (
    get_job, 
    list_jobs, 
    clean_old_jobs,
    serialize_job
)
from app.core.schemas.responses import JobStatusResponse

router = APIRouter()

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str = Path(..., description="Job ID")):
    """
    Get job status
    """
    job = get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return serialize_job(job)

@router.get("/jobs", response_model=List[JobStatusResponse])
async def list_all_jobs(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of jobs to return"),
    skip: int = Query(0, ge=0, description="Number of jobs to skip"),
):
    """
    List jobs with pagination
    """
    jobs = list_jobs(limit=limit, skip=skip)
    return [serialize_job(job) for job in jobs]

@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str = Path(..., description="Job ID")):
    """
    Delete job
    """
    from app.core.services.job_storage import JOB_STORAGE
    
    if job_id not in JOB_STORAGE:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    del JOB_STORAGE[job_id]
    
    return {"message": f"Job {job_id} deleted"}

@router.post("/jobs/cleanup")
async def clean_jobs(max_age_hours: int = Query(24, ge=1, description="Maximum age of jobs in hours")):
    """
    Clean up old jobs
    """
    count = clean_old_jobs(max_age_hours=max_age_hours)
    
    return {"message": f"Cleaned {count} old jobs"}