"""Main package for LLM-based review inference for Lamoda."""

from .clients import LLMClient
from .config import Config
from .core import run_inference, run_pipeline_for_file, run_pipeline_for_sku
from .utils import (
    build_prompt,
    execute_llm,
    get_golden_tags_for_product,
    load_dataset,
    load_golden_tags_from_dict,
    postprocess_tags,
    prepare_reviews,
)

__version__ = "0.1.0"

# Try to import OpenAI client if available
try:
    from .clients import OpenAIClient
    _openai_available = True
except ImportError:
    _openai_available = False

# Build __all__ dynamically
__all__ = [
    # Config
    "Config",
    # Core functions
    "run_inference",
    "run_pipeline_for_file",
    "run_pipeline_for_sku",
    # Clients
    "LLMClient",
    # Utility functions
    "load_dataset",
    "load_golden_tags_from_dict",
    "prepare_reviews",
    "build_prompt",
    "get_golden_tags_for_product",
    "execute_llm",
    "postprocess_tags",
]

# Add OpenAI client to exports if available
if _openai_available:
    __all__.append("OpenAIClient")
