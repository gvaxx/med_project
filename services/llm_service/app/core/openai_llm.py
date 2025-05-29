from typing import Dict, Any
import openai
from .llm_base import LLMBase

class OpenAILLM(LLMBase):
    """OpenAI LLM implementation."""
    
    # def __init__(self, api_key: str, model: str = "gpt-4o-mini-2024-07-18"):
    def __init__(self, api_key: str, model: str = "gpt-4o-2024-11-20"):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            http_client=None  # Let OpenAI handle the HTTP client configuration
        )
        self.model = model
    
    async def generate_response(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating response from OpenAI: {str(e)}")
    
    async def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model,
            "type": "openai",
            "is_local": False
        } 