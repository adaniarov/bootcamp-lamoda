"""Example of using pipeline for file processing."""

import logging
from typing import Dict, List

from src.clients import LLMClient
from src.core import run_pipeline_for_file, run_pipeline_for_sku
from .mock_client import MockLLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_run_pipeline_for_file():
    """Example of running pipeline for entire file."""
    # Create mock LLM client
    llm_client: LLMClient = MockLLMClient()

    # Prepare GOLDEN TAGS dictionaries
    name_to_tags: Dict[str, List[str]] = {
        "T-Shirt": ["quality", "size", "material", "price", "color"],
        "Jeans": ["quality", "size", "fit", "price"],
        "Socks": ["quality", "size", "material"],
    }

    subtype_to_tags: Dict[str, List[str]] = {
        "TEE-SHIRTS & POLOS": ["quality", "size", "material"],
        "JEANS": ["quality", "size", "fit"],
        "SOCKS & TIGHTS": ["quality", "size"],
    }

    type_to_tags: Dict[str, List[str]] = {
        "Clothes": ["quality", "size", "price"],
        "Shoes": ["quality", "size", "comfort"],
    }

    # Run pipeline for file
    results_df = run_pipeline_for_file(
        csv_path="data/raw/lamoda_reviews.csv",
        llm_client=llm_client,
        name_to_tags=name_to_tags,
        subtype_to_tags=subtype_to_tags,
        type_to_tags=type_to_tags,
        output_path="data/processed/llm_tags_results.csv",
        max_chars=500,
        max_reviews=50,
        min_review_length=10,
        max_tags=6,
        limit_skus=10,  # Limit for testing
        skip_errors=True,
    )

    logger.info(f"\nProcessed {len(results_df)} SKUs")
    logger.info(f"SKUs with tags: {len(results_df[results_df['num_tags'] > 0])}")
    logger.info("\nFirst results:")
    logger.info(f"\n{results_df.head()}")


def example_run_pipeline_for_sku():
    """Example of running pipeline for one SKU."""
    # Create mock LLM client
    llm_client: LLMClient = MockLLMClient()

    # Prepare GOLDEN TAGS dictionaries
    name_to_tags: Dict[str, List[str]] = {
        "T-Shirt": ["quality", "size", "material"],
    }

    # Run pipeline for one SKU
    tags = run_pipeline_for_sku(
        csv_path="data/raw/lamoda_reviews.csv",
        sku="MP002XW0O0SI",  # Example SKU
        llm_client=llm_client,
        name_to_tags=name_to_tags,
        max_chars=500,
        max_reviews=50,
        max_tags=6,
    )

    logger.info(f"\nReceived tags for SKU: {tags}")


if __name__ == "__main__":
    logger.info("=== Example 1: Process entire file ===")
    example_run_pipeline_for_file()

    logger.info("\n=== Example 2: Process one SKU ===")
    example_run_pipeline_for_sku()

