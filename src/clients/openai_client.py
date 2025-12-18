"""OpenAI LLM client implementation."""

import logging
import os
from typing import Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


class OpenAIClient:
    """LLM client for working with OpenAI API.
    
    This client provides a clean interface to OpenAI's chat completion API,
    with support for retries, error handling, and configurable parameters.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 200,
        store: bool = True,
    ):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
            model: Model name. Default: "gpt-4o-mini".
            temperature: Generation temperature (0.0-2.0). Default: 0.3.
            max_tokens: Maximum tokens in response. Default: 200.
            store: Whether to store requests in OpenAI history. Default: True.
            
        Raises:
            ImportError: If openai library is not installed.
            ValueError: If API key is not provided and not found in environment.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "The openai library is not installed. Install it with: poetry add openai"
            )
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter during initialization."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.store = store
        
        logger.info(
            f"OpenAI client initialized with model={model}, "
            f"temperature={temperature}, max_tokens={max_tokens}"
        )

    def generate(self, prompt: str) -> str:
        """Generate a response based on the prompt using OpenAI API.

        Args:
            prompt: Text prompt for the LLM.

        Returns:
            Response from LLM as a string.

        Raises:
            ValueError: If prompt is empty.
            Exception: If there's an error calling OpenAI API.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            logger.debug(f"Calling OpenAI API with prompt length: {len(prompt)}")
            
            completion = self.client.chat.completions.create(
                model=self.model,
                store=self.store,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant for analyzing product reviews. "
                            "Your task is to select relevant tags from the provided list."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            response = completion.choices[0].message.content
            if not response:
                logger.warning("Received empty response from OpenAI API")
                return ""

            logger.debug(f"Received response of length: {len(response)}")
            return response.strip()

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise Exception(f"Error calling OpenAI API: {e}") from e

