# Audio Transcription & Translation API

An API service for transcribing audio in English and Kinyarwanda, with bidirectional translation and caption generation capabilities.

## Features

- **Multilingual Transcription**: Support for English and Kinyarwanda audio files
- **Multiple Transcription Models**:
  - OpenAI Whisper (English)
  - Whisper-Kinyarwanda (specialized for Kinyarwanda)
  - NeMo (high accuracy for Kinyarwanda)
- **Bidirectional Translation**:
  - Kinyarwanda → English
  - English → Kinyarwanda
- **Caption Generation**:
  - SRT captions for video subtitling
  - WebVTT captions for web videos
  - Bilingual caption support
- **Full Processing Pipeline**:
  - Comprehensive transcription → translation → caption workflow
  - Asynchronous processing with job status tracking
  - Smart chunking for better timestamps

## System Requirements

### Minimum Requirements

- CPU: 4 cores (8 threads)
- RAM: 8GB
- Storage: 50GB SSD
- NVIDIA GPU with 8GB+ VRAM (for optimal performance)

### Recommended Requirements

- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD
- NVIDIA GPU with 16GB+ VRAM (T4, RTX 3080, or better)
- High-speed internet connection

## Installation & Setup

### Option 1: Direct Installation

#### Prerequisites

- Ubuntu 20.04 LTS or later
- Python 3.8+ (3.9 recommended)
- NVIDIA drivers and CUDA (for GPU acceleration)

#### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/yuyunfrancis/asr_api_kin_to_en.git
cd asr_api_kin_to_en
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Make sure you have nemo toolkit installed

```bash
sudo apt-get install build-essential python3-dev python3-pip
pip install "nemo_toolkit[all]"
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables by creating a `.env` file:

```
# API Settings
DEBUG=True

# Model Settings
MODELS_CACHE_DIR=./models_cache
HUGGING_FACE_TOKEN=your_huggingface_token_here

# File Storage
TEMP_FILE_DIR=./temp_files
TEMP_FILE_EXPIRY=24  # hours

# Resource Management
USE_GPU=True
MAX_AUDIO_SIZE=52428800  # 50MB
```

5. Create necessary directories:

```bash
mkdir -p models_cache temp_files
```

### Option 2: Docker Installation

#### Prerequisites

- Docker and Docker Compose
- NVIDIA Container Toolkit (for GPU support)

#### Installation Steps

6. Install Redis server

### Install Redis if not already installed

- sudo apt update
- sudo apt install redis-server

### Start Redis service

- sudo systemctl start redis

1. Clone the repository:

```bash
git clone https://github.com/yuyunfrancis/asr_api_kin_to_en.git
cd asr_api_kin_to_en
```

2. Create necessary directories:

```bash
mkdir -p models_cache temp_files
```

3. Create a `.env` file with your configuration (see example above)

4. Create or modify `docker-compose.yml`:

```yaml
version: "3.8"

services:
  api:
    build: .
    restart: always
    ports:
      - "8000:8000"
    volumes:
      - ./models_cache:/app/models_cache
      - ./temp_files:/app/temp_files
    environment:
      - HUGGING_FACE_TOKEN=${HUGGING_FACE_TOKEN}
      - MODELS_CACHE_DIR=/app/models_cache
      - TEMP_FILE_DIR=/app/temp_files
      - USE_GPU=True
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:6.2-alpine
    restart: always
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  nginx:
    image: nginx:1.21-alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - api

volumes:
  redis-data:
```

5. Create an `nginx.conf` file:

```nginx
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        client_max_body_size 50M;
        proxy_read_timeout 300s;
    }
}
```

## Running the Application

### Option 1: Running with Uvicorn (Development)

To run the application directly with Uvicorn:

```bash
# Activate virtual environment if not already activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run with Uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

This command:

- Starts the FastAPI application
- Enables hot reloading for development
- Makes the API accessible from other devices on the network
- Uses port 8000

### Option 2: Running with Docker Compose (Production)

To run the application using Docker Compose:

```bash
# Build and start the containers
docker-compose up -d

# View logs
docker-compose logs -f
```

The API will be accessible at `http://localhost:8000` or `http://localhost` if using the Nginx configuration.

## API Usage

### Interactive Documentation

Once the API is running, you can access the interactive API documentation:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Main Endpoints

| Endpoint                 | Method | Description                                               |
| ------------------------ | ------ | --------------------------------------------------------- |
| `/api/process-audio`     | POST   | Complete pipeline: transcription → translation → captions |
| `/api/transcribe`        | POST   | Audio transcription only                                  |
| `/api/translate`         | POST   | Text translation between languages                        |
| `/api/generate-captions` | POST   | Generate captions from transcription results              |
| `/api/jobs/{job_id}`     | GET    | Check the status of a job                                 |

### Example: Processing Audio (Complete Pipeline)

Using curl:

```bash
curl -X POST "http://localhost:8000/api/process-audio" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/audio.mp3" \
  -F "source_language=english" \
  -F "transcription_model=whisper" \
  -F "translation_options={\"enabled\":true,\"target_language\":\"kinyarwanda\"}" \
  -F "caption_options={\"enabled\":true,\"formats\":[\"srt\",\"vtt\"]}"
```

Using JavaScript fetch:

```javascript
async function processAudio(audioFile) {
  const formData = new FormData();
  formData.append("file", audioFile);
  formData.append("source_language", "english");
  formData.append("transcription_model", "whisper");
  formData.append(
    "translation_options",
    JSON.stringify({
      enabled: true,
      target_language: "kinyarwanda",
    })
  );
  formData.append(
    "caption_options",
    JSON.stringify({
      enabled: true,
      formats: ["srt", "vtt"],
    })
  );

  const response = await fetch("http://localhost:8000/api/process-audio", {
    method: "POST",
    body: formData,
  });

  const job = await response.json();
  return job.job_id; // Use this to check job status
}
```

### Checking Job Status

Using curl:

```bash
curl -X GET "http://localhost:8000/api/jobs/YOUR_JOB_ID" -H "accept: application/json"
```

Using JavaScript fetch:

```javascript
async function checkJobStatus(jobId) {
  const response = await fetch(`http://localhost:8000/api/jobs/${jobId}`);
  return await response.json();
}
```

Sample parameters to test

```bash
{"chunk_size": 10, "overlap_size": 5, "use_mfa": false}

{"enabled": true, "target_language": "kinyarwanda"}

{"enabled": true, "formats": ["srt", "vtt"]}
```

## Performance Optimization

### Model Preloading

For faster initial response times, you can "warm up" the models by making an initial request for each language you plan to support before heavy usage.

### GPU Memory Management

If you encounter GPU memory issues:

1. Adjust the batch size and chunk parameters in your requests
2. Consider using a GPU with more VRAM
3. Add more worker processes for processing multiple files concurrently

### File Management

The API automatically cleans up temporary files, but for high-volume deployments you may want to:

1. Mount a larger volume for the temp_files directory
2. Adjust the TEMP_FILE_EXPIRY setting in .env
3. Set up a cron job to periodically clean old files

## Troubleshooting

### Common Issues

1. **Model loading errors**:

   - Ensure your HUGGING_FACE_TOKEN is valid
   - Check if you have internet access to download models
   - Verify there's enough disk space in the models_cache directory

2. **GPU not detected**:

   - Check NVIDIA drivers with `nvidia-smi`
   - For Docker: ensure nvidia-container-toolkit is properly configured
   - Try setting USE_GPU=False to use CPU mode

3. **Empty caption files**:
   - This is a known issue with Kinyarwanda → English translation
   - Check the chunk structure in the translation process
   - Ensure both languages have the correct field names

### Logs

- Direct installation: Check console output
- Docker installation: `docker-compose logs -f api`

## Maintenance and Updates

### Updating the API

For direct installation:

```bash
git pull
source venv/bin/activate
pip install -r requirements.txt
```

For Docker installation:

```bash
git pull
docker-compose down
docker-compose up -d --build
```

### Adding New Languages

The API is designed to support additional languages by:

1. Adding new models in the `app/config.py` file
2. Updating the `LANGUAGE_CODES` mapping
3. Implementing language-specific processing in the services

## Acknowledgments

- [MbazaNLP](https://huggingface.co/mbazaNLP) for the Kinyarwanda models
- [OpenAI](https://github.com/openai/whisper) for the Whisper model
- [NVIDIA](https://github.com/NVIDIA/NeMo) for the NeMo toolkit
- [Hugging Face](https://huggingface.co/) for model hosting and transformers library

## Contact

For support or inquiries, please contact [francisberi04@gmail.com].
