"""Основной модуль для работы с LLM инференсом отзывов Lamoda."""

from .data_loader import load_dataset, load_golden_tags, load_golden_tags_from_dict
from .inference import run_inference
from .llm_client import LLMClient, BaseLLMClient
from .llm_inference import run_llm
from .openai_client import OpenAILLMClient
from .pipeline import run_pipeline_for_file, run_pipeline_for_sku
from .postprocessing import postprocess_tags
from .preprocessing import prepare_reviews
from .prompt_builder import build_prompt, get_golden_tags_for_product

__all__ = [
    "run_inference",
    "run_pipeline_for_file",
    "run_pipeline_for_sku",
    "load_dataset",
    "load_golden_tags",
    "load_golden_tags_from_dict",
    "prepare_reviews",
    "build_prompt",
    "get_golden_tags_for_product",
    "run_llm",
    "postprocess_tags",
    "LLMClient",
    "BaseLLMClient",
    "OpenAILLMClient",
]

