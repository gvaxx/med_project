from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMBase(ABC):
    """Abstract base class for LLM models."""
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM model.
        
        Args:
            prompt (str): The input prompt for the model
            **kwargs: Additional model-specific parameters
            
        Returns:
            str: The generated response
        """
        pass
    
    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dict[str, Any]: Information about the model including name, type, etc.
        """
        pass 