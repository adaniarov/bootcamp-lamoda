"""LLM clients package."""

from .llm_client import LLMClient

# Try to import OpenAI client, but don't fail if openai is not installed
try:
    from .openai_client import OpenAIClient
    __all__ = ["LLMClient", "OpenAIClient"]
except ImportError:
    # OpenAI library not installed, only export LLMClient
    __all__ = ["LLMClient"]

