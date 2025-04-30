from typing import Dict, Any
import httpx
from .llm_base import LLMBase
import logging
import os

logger = logging.getLogger(__name__)

class DeepSeekLLM(LLMBase):
    """DeepSeek LLM implementation."""
    
    def __init__(self, api_key: str = None):
        """Initialize DeepSeek LLM with API configuration."""
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = os.getenv("DEEPSEEK_API_BASE", "http://localhost:8000")
        self.model = "deepseek-chat"
        
    async def generate_response(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Generate response using DeepSeek model."""
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:  # 10 minutes timeout
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = await client.post(
                    f"{self.api_base}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": kwargs.get("temperature", 0.7),
                        "max_tokens": kwargs.get("max_tokens", 4000),
                        "top_p": kwargs.get("top_p", 0.95),
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.error(f"Error generating response from DeepSeek: {str(e)}")
            raise Exception(f"Error generating response from DeepSeek: {str(e)}")
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the DeepSeek model."""
        return {
            "name": self.model,
            "type": "deepseek",
            "is_local": False,
            "api_base": self.api_base
        } 