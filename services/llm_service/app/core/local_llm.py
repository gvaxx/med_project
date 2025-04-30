from typing import Dict, Any
import os
import httpx
import logging
import asyncio
from .llm_base import LLMBase

logger = logging.getLogger(__name__)

class LocalLLM(LLMBase):
    """Local LLM implementation using LM Studio server."""
    
    def __init__(self, model_path: str):
        """
        Initialize LocalLLM with LM Studio configuration.
        
        Args:
            model_path: Path to the model (not used directly, as model is loaded in LM Studio)
        """
        self.model_path = model_path
        # LM Studio по умолчанию работает на порту 1234
        self.api_base_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
        logger.info(f"Initializing LocalLLM with API URL: {self.api_base_url}")
    
    async def _check_lm_studio_health(self) -> bool:
        """Проверяет доступность LM Studio."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.api_base_url}/models")
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"LM Studio health check failed: {str(e)}")
            return False

    async def generate_response(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """
        Generate response using LM Studio's local model.
        
        Args:
            prompt: The user's input prompt
            system_prompt: Optional system prompt to guide the model's behavior
            **kwargs: Additional parameters for generation (temperature, max_tokens, etc.)
        
        Returns:
            str: Generated response from the model
        """
        # Проверяем доступность LM Studio перед отправкой запроса
        if not await self._check_lm_studio_health():
            raise Exception("LM Studio is not available. Please check if it's running and accessible.")

        try:
            # Используем более длительный таймаут для чтения
            timeout = httpx.Timeout(
                connect=10.0,    # таймаут подключения
                read=600.0,      # увеличенный таймаут чтения (10 минут)
                write=30.0,      # таймаут записи
                pool=10.0        # таймаут пула
            )
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                params = {
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "stream": False,
                    # Добавляем параметры для управления генерацией
                    "top_p": kwargs.get("top_p", 0.95),
                    "presence_penalty": kwargs.get("presence_penalty", 0.0),
                    "frequency_penalty": kwargs.get("frequency_penalty", 0.0)
                }
                
                logger.debug(f"Sending request to LM Studio with params: {params}")
                
                try:
                    # Добавляем повторные попытки при ошибках
                    max_retries = 3
                    retry_delay = 1
                    last_error = None
                    
                    for attempt in range(max_retries):
                        try:
                            response = await client.post(
                                f"{self.api_base_url}/chat/completions",
                                json=params,
                                headers={"Content-Type": "application/json"}
                            )
                            response.raise_for_status()
                            
                            result = response.json()
                            logger.debug(f"Received response from LM Studio: {result}")
                            
                            if not result.get("choices"):
                                raise ValueError("No choices in LM Studio response")
                            
                            if not result["choices"][0].get("message"):
                                raise ValueError("No message in LM Studio response choice")
                            
                            content = result["choices"][0]["message"].get("content")
                            if not content:
                                raise ValueError("No content in LM Studio response message")
                            
                            return content
                            
                        except (httpx.HTTPStatusError, httpx.ReadTimeout) as e:
                            last_error = e
                            if attempt == max_retries - 1:  # Последняя попытка
                                break
                            logger.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Увеличиваем задержку между попытками
                    
                    # Если все попытки не удались, выбрасываем последнюю ошибку
                    if last_error:
                        if isinstance(last_error, httpx.HTTPStatusError):
                            logger.error(f"HTTP error from LM Studio: {last_error.response.status_code} - {last_error.response.text}")
                            raise Exception(f"LM Studio returned error {last_error.response.status_code}: {last_error.response.text}")
                        elif isinstance(last_error, httpx.ReadTimeout):
                            logger.error("Timeout while waiting for LM Studio response")
                            raise Exception("LM Studio response timeout. The model might be busy or the request is too complex.")
                    
                except httpx.RequestError as e:
                    logger.error(f"Error making request to LM Studio: {str(e)}")
                    raise Exception(f"Failed to connect to LM Studio: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}", exc_info=True)
            raise Exception(f"Error generating response from local model: {str(e)}")
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model in LM Studio."""
        try:
            # Используем общий таймаут в 10 секунд для всех операций
            async with httpx.AsyncClient(timeout=10.0) as client:
                try:
                    response = await client.get(f"{self.api_base_url}/models")
                    response.raise_for_status()
                    models_info = response.json()
                    
                    logger.debug(f"Retrieved model info from LM Studio: {models_info}")
                    
                    return {
                        "name": "lm_studio_model",
                        "type": "local",
                        "is_local": True,
                        "model_path": self.model_path,
                        "api_base": self.api_base_url,
                        "available_models": models_info
                    }
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error getting model info: {e.response.status_code} - {e.response.text}")
                    raise
                except httpx.RequestError as e:
                    logger.error(f"Error connecting to LM Studio for model info: {str(e)}")
                    raise
        except Exception as e:
            logger.error(f"Error in get_model_info: {str(e)}", exc_info=True)
            return {
                "name": "lm_studio_model",
                "type": "local",
                "is_local": True,
                "model_path": self.model_path,
                "api_base": self.api_base_url,
                "error": str(e)
            } 