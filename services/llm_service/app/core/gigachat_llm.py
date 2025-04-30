from typing import Dict, Any
import logging
import os
from .llm_base import LLMBase
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

logger = logging.getLogger(__name__)

class GigaChatLLM(LLMBase):
    """GigaChat LLM implementation using native API."""
    
    def __init__(self, credentials: str = None):
        """Initialize GigaChat LLM with API configuration."""
        self.credentials = credentials or os.getenv("GIGACHAT_CREDENTIALS")
        self.scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.model = os.getenv("GIGACHAT_MODEL", "GigaChat")
        
        if not self.credentials:
            raise ValueError("GigaChat credentials not provided")
            
        logger.info("Initializing GigaChat with scope: %s, model: %s", self.scope, self.model)
    
    async def generate_response(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Generate response using GigaChat model."""
        try:
            # Create chat payload
            payload = Chat(
                messages=[],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1024)
            )
            
            # Add system prompt if provided
            if system_prompt:
                payload.messages.append(
                    Messages(
                        role=MessagesRole.SYSTEM,
                        content=system_prompt
                    )
                )
            
            # Add user prompt
            payload.messages.append(
                Messages(
                    role=MessagesRole.USER,
                    content=prompt
                )
            )
            
            # Generate response using context manager
            logger.info(f"GigaChat credentials: {self.credentials}")
            with GigaChat(
                credentials=self.credentials,
                verify_ssl_certs=False,
                scope=self.scope,
                model=self.model
            ) as giga:
                response = giga.chat(payload)
                return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response from GigaChat: {str(e)}")
            raise Exception(f"Error generating response from GigaChat: {str(e)}")
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the GigaChat model."""
        try:
            with GigaChat(
                credentials=self.credentials,
                verify_ssl_certs=False,
                scope=self.scope,
                model=self.model
            ) as giga:
                models = giga.get_models()
                # Convert models to a simple dictionary
                models_info = {
                    "current_model": self.model,
                    "available_models": [
                        {
                            "id": model.id,
                            "name": model.name,
                            "family": model.family
                        } for model in models.models
                    ] if hasattr(models, 'models') else []
                }
                return {
                    "name": self.model,
                    "type": "gigachat",
                    "is_local": False,
                    "scope": self.scope,
                    "models_info": models_info
                }
        except Exception as e:
            logger.error(f"Error getting GigaChat model info: {str(e)}")
            return {
                "name": self.model,
                "type": "gigachat",
                "is_local": False,
                "scope": self.scope,
                "error": str(e)
            } 