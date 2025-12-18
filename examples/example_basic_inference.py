"""Example of using the inference function."""

import logging
from typing import List

from src.clients import LLMClient
from src.core import run_inference
from .mock_client import MockLLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_usage():
    """Example of using the run_inference function."""
    # Create mock LLM client
    llm_client: LLMClient = MockLLMClient()

    # Example data
    reviews: List[str] = [
        "Excellent product quality, very satisfied with the purchase!",
        "Good size, fits perfectly.",
        "Quality is excellent, recommend.",
        "Size is a bit small, but quality is good.",
    ]

    # Dictionaries for GOLDEN TAGS
    name_to_tags = {
        "T-Shirt": ["quality", "size", "material", "price", "color"],
    }

    subtype_to_tags = {
        "TEE-SHIRTS & POLOS": ["quality", "size", "material"],
    }

    type_to_tags = {
        "Clothes": ["quality", "size", "price"],
    }

    # Execute inference
    tags = run_inference(
        reviews=reviews,
        llm_client=llm_client,
        name_to_tags=name_to_tags,
        subtype_to_tags=subtype_to_tags,
        type_to_tags=type_to_tags,
        product_name="T-Shirt",
        product_subtype="TEE-SHIRTS & POLOS",
        product_type="Clothes",
        max_chars=500,
        max_reviews=50,
        min_review_length=10,
        max_tags=6,
    )

    logger.info(f"Result: {tags}")
    return tags


if __name__ == "__main__":
    example_usage()

