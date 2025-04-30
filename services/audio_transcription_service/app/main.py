import os
import shutil
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import uuid
import gigaam
from pathlib import Path
import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Audio Transcription Service")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize GigaAM models
try:
    logger.info("Initializing GigaAM models...")
    
    # Проверка наличия зависимостей для longform транскрипции
    try:
        import importlib
        longform_deps = ["pyannote.audio"]
        missing_deps = []
        
        for dep in longform_deps:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            logger.warning(f"Missing longform dependencies: {', '.join(missing_deps)}")
            logger.warning("Long-form transcription may not work properly")
        else:
            logger.info("All longform dependencies are available")
    except Exception as e:
        logger.warning(f"Error checking longform dependencies: {str(e)}")
    
    # Based on the GigaAM documentation, v2_rnnt is the best model with lowest WER
    # Available model names: "v2_ctc" or "ctc", "v2_rnnt" or "rnnt", "v1_ctc", "v1_rnnt"
    
    # First try to load the best models (v2)
    try:
        logger.info("Attempting to load GigaAM-v2 RNNT model (best performance)")
        rnnt_model = gigaam.load_model("v2_rnnt")
        logger.info("Successfully loaded GigaAM-v2 RNNT model")
    except Exception as e:
        logger.warning(f"Failed to load v2_rnnt model: {str(e)}. Falling back to default RNNT model.")
        try:
            rnnt_model = gigaam.load_model("rnnt")
            logger.info("Successfully loaded default RNNT model")
        except Exception as e2:
            logger.warning(f"Failed to load default RNNT model: {str(e2)}. Trying v1_rnnt.")
            try:
                rnnt_model = gigaam.load_model("v1_rnnt")
                logger.info("Successfully loaded GigaAM-v1 RNNT model")
            except Exception as e3:
                logger.error(f"Failed to load any RNNT model: {str(e3)}")
                rnnt_model = None
    
    # Load CTC model as fallback
    try:
        logger.info("Attempting to load GigaAM-v2 CTC model")
        ctc_model = gigaam.load_model("v2_ctc")
        logger.info("Successfully loaded GigaAM-v2 CTC model")
    except Exception as e:
        logger.warning(f"Failed to load v2_ctc model: {str(e)}. Falling back to default CTC model.")
        try:
            ctc_model = gigaam.load_model("ctc")
            logger.info("Successfully loaded default CTC model")
        except Exception as e2:
            logger.warning(f"Failed to load default CTC model: {str(e2)}. Trying v1_ctc.")
            try:
                ctc_model = gigaam.load_model("v1_ctc")
                logger.info("Successfully loaded GigaAM-v1 CTC model")
            except Exception as e3:
                logger.error(f"Failed to load any CTC model: {str(e3)}")
                ctc_model = None
    
    # Log which models were successfully loaded
    if rnnt_model is not None:
        logger.info("RNNT model is available")
    if ctc_model is not None:
        logger.info("CTC model is available")
    if rnnt_model is None and ctc_model is None:
        logger.error("Failed to load any GigaAM models")
        
except Exception as e:
    logger.error(f"Error during GigaAM model initialization: {str(e)}")
    ctc_model = None
    rnnt_model = None

class TranscriptionRequest(BaseModel):
    model_type: str = "rnnt"  # "ctc" or "rnnt"
    long_form: bool = False

class TranscriptionResponse(BaseModel):
    id: str
    filename: str
    transcription: str
    model_type: str
    duration: Optional[float] = None

class LongFormUtterance(BaseModel):
    transcription: str
    boundaries: List[float]  # [start_time, end_time]

class LongFormTranscriptionResponse(BaseModel):
    id: str
    filename: str
    utterances: List[LongFormUtterance]
    model_type: str
    duration: Optional[float] = None
    transcription: str


@app.get("/")
async def root():
    return {"message": "Audio Transcription Service API"}

@app.get("/health")
async def health_check():
    if ctc_model is None and rnnt_model is None:
        return {
            "status": "unhealthy",
            "message": "GigaAM models failed to load"
        }
    return {"status": "healthy"}

@app.get("/models")
async def get_available_models():
    models = {
        "ctc": ctc_model is not None,
        "rnnt": rnnt_model is not None
    }
    return models

@app.post("/debug-upload")
async def debug_upload(file: UploadFile = File(...)):
    """Debug endpoint to test file upload functionality"""
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(await file.read()),
        "headers": dict(file.headers)
    }

def cleanup_file(file_path: str):
    """Delete temporary audio file after processing"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    model_type: str = Form("rnnt"),
    long_form: str = Form("false")
):
    """Simplified endpoint that takes form data directly"""
    logger.info(f"Received request with model_type={model_type}, long_form={long_form}")
    logger.info(f"File: {file.filename}, Content-Type: {file.content_type}")
    logger.info(f"long_form: {long_form}")
    # Convert long_form to boolean
    use_long_form = long_form.lower() == "true"
    logger.info(f"use_long_form: {use_long_form}")
    # Validate model type
    if model_type not in ["ctc", "rnnt"]:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid model_type: {model_type}. Must be 'ctc' or 'rnnt'"}
        )
    
    # Get model
    model = ctc_model if model_type == "ctc" else rnnt_model
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"error": f"Model {model_type} is not available"}
        )
    
    # Save file
    temp_dir = None
    temp_file_path = None
    
    try:
        temp_dir = tempfile.mkdtemp()
        file_extension = os.path.splitext(file.filename)[1] or ".wav"
        temp_file_path = os.path.join(temp_dir, f"audio{file_extension}")
        
        # Read file content and save to disk
        file_content = await file.read()
        if not file_content:
            return JSONResponse(
                status_code=400,
                content={"error": "File is empty"}
            )
        
        # Write to disk
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"Saved file to {temp_file_path}")
        
        # Process based on long_form
        if not use_long_form:
            # Short audio transcription
            process_short_audio = True
        else:
            # Try long-form transcription first
            process_short_audio = False
            
            # Set Hugging Face token if available
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
                logger.info("Using HF_TOKEN for longform transcription")
            
            try:
                utterances = model.transcribe_longform(temp_file_path)
                transcription = " ".join([u["transcription"] for u in utterances])
                return {
                    "utterances": [
                        {
                            "transcription": u["transcription"],
                            "boundaries": u["boundaries"]
                        }
                        for u in utterances
                    ],
                    "transcription": transcription,
                    "model_type": model_type,
                    "file_info": {
                        "filename": file.filename,
                        "size": len(file_content),
                        "content_type": file.content_type
                    }
                }
            except Exception as e:
                logger.error(f"Long-form transcription error: {str(e)}")
                logger.info("Falling back to regular transcription")
                # Fallback to short audio transcription
                process_short_audio = True
        
        # Process short audio if needed
        if process_short_audio:
            transcription = model.transcribe(temp_file_path)
            
            return {
                "transcription": transcription,
                "model_type": model_type,
                "file_info": {
                    "filename": file.filename,
                    "size": len(file_content),
                    "content_type": file.content_type
                }
            }
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"File processing error: {str(e)}"}
        )
    finally:
        # Cleanup
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")
            pass

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify the service is responsive"""
    return {
        "status": "ok",
        "models": {
            "ctc": ctc_model is not None,
            "rnnt": rnnt_model is not None
        },
        "timestamp": datetime.datetime.now().isoformat()
    } 