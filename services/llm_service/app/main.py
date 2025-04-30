from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.core.openai_llm import OpenAILLM
from app.core.local_llm import LocalLLM
from app.core.deepseek_llm import DeepSeekLLM
from app.core.gigachat_llm import GigaChatLLM
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Service")

# Model configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "models/local_model")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")

logger.info(f"Initializing LLM Service with available models...")

# LLM instances
llm_instances = {
    "openai": OpenAILLM(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None,
    "local": LocalLLM(model_path=LOCAL_MODEL_PATH),
    "deepseek": DeepSeekLLM(api_key=DEEPSEEK_API_KEY) if DEEPSEEK_API_KEY else None,
    "gigachat": GigaChatLLM(credentials=GIGACHAT_CREDENTIALS) if GIGACHAT_CREDENTIALS else None
}

# Log available models
for model_type, instance in llm_instances.items():
    if instance is not None:
        logger.info(f"Model {model_type} is available")
    else:
        logger.warning(f"Model {model_type} is not configured")

class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str = ""
    model_type: str = "openai"  # "openai", "local", "deepseek", or "gigachat"
    parameters: Optional[Dict[str, Any]] = None

class GenerateResponse(BaseModel):
    response: str
    model_info: Dict[str, Any]

@app.get("/")
async def root():
    return {"message": "LLM Service API"}

@app.get("/health")
async def health_check():
    try:
        available_models = {}
        for model_type, instance in llm_instances.items():
            if instance is not None:
                try:
                    model_info = await instance.get_model_info()
                    available_models[model_type] = {"status": "available", "info": model_info}
                except Exception as e:
                    available_models[model_type] = {"status": "error", "error": str(e)}
            else:
                available_models[model_type] = {"status": "not_configured"}
        
        return {
            "status": "healthy",
            "models": available_models
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/available_models")
async def get_available_models():
    available = {}
    for model_type, instance in llm_instances.items():
        if instance is not None:
            try:
                model_info = await instance.get_model_info()
                available[model_type] = model_info
            except Exception as e:
                logger.error(f"Error getting model info for {model_type}: {str(e)}", exc_info=True)
    return available

@app.post("/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    logger.info(f"Received generation request for model_type: {request.model_type}")
    logger.debug(f"Request parameters: {request.parameters}")
    
    if request.model_type not in llm_instances:
        logger.error(f"Unknown model type requested: {request.model_type}")
        raise HTTPException(status_code=400, detail=f"Unknown model type: {request.model_type}")
    
    llm = llm_instances[request.model_type]
    if llm is None:
        logger.error(f"Model type {request.model_type} is not configured")
        raise HTTPException(
            status_code=400, 
            detail=f"Model type {request.model_type} is not configured properly"
        )
    
    try:
        parameters = request.parameters or {}
        logger.info("Generating response...")
        response = await llm.generate_response(request.prompt, request.system_prompt, **parameters)
        logger.info("Successfully generated response")
        logger.debug(f"Generated response: {response[:100]}...")  # Log first 100 chars
        
        model_info = await llm.get_model_info()
        logger.debug(f"Model info: {model_info}")
        
        return GenerateResponse(
            response=response,
            model_info=model_info
        )
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 