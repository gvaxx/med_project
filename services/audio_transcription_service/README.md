# Audio Transcription Service

This service provides audio transcription capabilities using GigaAM-v2 models, which are state-of-the-art speech recognition models for the Russian language.

## Features

- **Audio Transcription**: Convert speech to text using GigaAM-v2 models
- **Multiple Model Options**: Choose between CTC and RNNT models
- **Long-form Transcription**: Support for transcribing longer audio files with automatic segmentation
- **REST API**: Simple API for integration with other services

## Architecture

The service is built using FastAPI and exposes a simple REST API for transcribing audio files. It uses GigaAM-v2 models from Salute Developers for speech recognition.

### API Endpoints

- `GET /health`: Health check endpoint
- `GET /models`: Get available transcription models
- `POST /transcribe`: Transcribe an audio file

## Models

The service uses the following GigaAM-v2 models:

- **GigaAM-CTC-v2**: A model using Connectionist Temporal Classification (CTC) for speech recognition
- **GigaAM-RNNT-v2**: A model using RNN Transducer, which generally provides better accuracy

## Long-form Transcription

For audio files longer than 30 seconds, the service provides a "long-form" transcription option that:

1. Splits the audio into segments using Voice Activity Detection (VAD)
2. Transcribes each segment separately
3. Returns the transcription with time boundaries for each segment

## Requirements

- Python 3.9+
- ffmpeg (for audio processing)
- GigaAM Python package

## Installation for Development

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the service:
   ```
   uvicorn app.main:app --host 0.0.0.0 --port 8004
   ```

## Docker Deployment

The service can be deployed using Docker:

```
docker build -t audio-transcription-service .
docker run -p 8004:8004 audio-transcription-service
```

## How to Use

### Basic Transcription

Send a POST request to the `/transcribe` endpoint with the audio file and optional parameters:

```python
import requests

files = {'file': open('audio.mp3', 'rb')}
data = {'model_type': 'rnnt', 'long_form': False}

response = requests.post('http://localhost:8004/transcribe', files=files, data=data)
print(response.json())
```

### Long-form Transcription

For longer audio files, set the `long_form` parameter to `True`:

```python
data = {'model_type': 'rnnt', 'long_form': True}
```

## Credits

This service is built on top of GigaAM-v2 models developed by Salute Developers. More information can be found at [https://github.com/salute-developers/GigaAM](https://github.com/salute-developers/GigaAM). 