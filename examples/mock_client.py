"""Mock LLM client for testing and examples."""

import logging

logger = logging.getLogger(__name__)


class MockLLMClient:
    """Example LLM client implementation for testing.
    
    This mock client doesn't make real API calls and is useful for:
    - Testing without incurring API costs
    - Development and debugging
    - Examples and demonstrations
    """

    def generate(self, prompt: str) -> str:
        """Generate a mock response based on the prompt.

        Args:
            prompt: Text prompt for LLM.

        Returns:
            Mock response from LLM as a string.
        """
        logger.debug("MockLLMClient: generating mock response")
        
        # Simple mock implementation for demonstration
        if "quality" in prompt.lower() or "качество" in prompt.lower():
            return "quality, size"
        
        return "tag1, tag2"

