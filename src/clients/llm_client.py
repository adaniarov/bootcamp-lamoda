"""Protocol for LLM client abstraction."""

from typing import Protocol


class LLMClient(Protocol):
    """Protocol for LLM client implementations.
    
    This protocol defines the interface that all LLM clients must implement.
    It allows for easy swapping between different LLM providers (OpenAI, Anthropic, etc.).
    """

    def generate(self, prompt: str) -> str:
        """Generate a response based on the prompt.

        Args:
            prompt: Text prompt for the LLM.

        Returns:
            Response from LLM as a string.

        Raises:
            Exception: If there's an error calling the LLM API.
        """
        ...

