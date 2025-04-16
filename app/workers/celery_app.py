from celery import Celery
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Redis connection settings from environment
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")

# Configure Celery
app = Celery(
    "audio_processing",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
    include=["app.workers.tasks"]
)

# Configure Celery settings
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Add a task result expiration time (3 days)
    result_expires=259200,
    # Add task routing if needed for different queues
    task_routes={
        "app.workers.tasks.process_transcription": {"queue": "transcription"},
        "app.workers.tasks.process_translation": {"queue": "translation"},
        "app.workers.tasks.process_captions": {"queue": "captions"},
        "app.workers.tasks.process_full_pipeline": {"queue": "pipeline"},
    },
    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=3600,  # 1 hour max per task
)

# If we need to do any initialization on startup
@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # Example: Add periodic cleanup task
    sender.add_periodic_task(
        60.0 * 60.0,  # Run every hour
        cleanup_old_files.s(),
        name="cleanup old files every hour",
    )
    
# Example periodic task
@app.task
def cleanup_old_files():
    """Cleanup old temporary files"""
    from app.utils.file_utils import cleanup_old_files as cleanup
    cleaned = cleanup(24)
    return f"Cleaned {cleaned} old files"

if __name__ == "__main__":
    app.start()