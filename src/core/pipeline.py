"""Module for running full review processing pipeline."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from src.utils import load_dataset, load_golden_tags_from_dict
from .tag_inference import run_inference

if TYPE_CHECKING:
    from src.clients import LLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline_for_file(
    csv_path: str,
    llm_client: "LLMClient",
    name_to_tags: Optional[Dict[str, List[str]]] = None,
    subtype_to_tags: Optional[Dict[str, List[str]]] = None,
    type_to_tags: Optional[Dict[str, List[str]]] = None,
    output_path: Optional[str] = None,
    max_chars: int = 500,
    max_reviews: int = 50,
    min_review_length: int = 10,
    max_tags: int = 6,
    min_reviews_per_sku: int = 1,
    custom_prompt_template: Optional[str] = None,
    skip_errors: bool = True,
    limit_skus: Optional[int] = None,
) -> pd.DataFrame:
    """Run full review processing pipeline for file.

    Executes the following steps:
    1. Load data from CSV
    2. Group reviews by SKU
    3. For each SKU run inference through LLM
    4. Save results to CSV (if output_path is specified)

    Args:
        csv_path: Path to CSV file with reviews.
        llm_client: Client for working with LLM.
        name_to_tags: Dictionary mapping name -> list of tags.
        subtype_to_tags: Dictionary mapping subtype -> list of tags.
        type_to_tags: Dictionary mapping type -> list of tags.
        output_path: Path to save results. If None, results are not saved.
        max_chars: Maximum review length in characters. Default: 500.
        max_reviews: Maximum number of reviews to process. Default: 50.
        min_review_length: Minimum review length to include. Default: 10.
        max_tags: Maximum number of tags to return. Default: 6.
        min_reviews_per_sku: Minimum number of reviews to include SKU. Default: 1.
        custom_prompt_template: Custom prompt template (optional).
        skip_errors: Skip errors for individual SKUs and continue processing. Default: True.
        limit_skus: Limit number of SKUs to process (for testing). Default: None.

    Returns:
        DataFrame with processing results. Columns: sku, name, subtype, type, 
        tags (string with comma-separated tags), num_tags, num_reviews, error (if there was an error).

    Raises:
        FileNotFoundError: If input file not found.
        ValueError: If data cannot be loaded.

    Examples:
        >>> class MockLLMClient:
        ...     def generate(self, prompt: str) -> str:
        ...         return "quality, size"
        >>> client = MockLLMClient()
        >>> name_tags = {"T-Shirt": ["quality", "size"]}
        >>> results = run_pipeline_for_file(
        ...     "data/raw/lamoda_reviews.csv",
        ...     client,
        ...     name_to_tags=name_tags,
        ...     limit_skus=5
        ... )
        >>> len(results) > 0
        True
    """
    logger.info(f"Starting file processing: {csv_path}")

    # Load data
    try:
        sku_data = load_dataset(
            csv_path=csv_path,
            min_reviews_per_sku=min_reviews_per_sku,
        )
        logger.info(f"Loaded {len(sku_data)} SKUs with reviews")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    # Load GOLDEN TAGS
    name_tags, subtype_tags, type_tags = load_golden_tags_from_dict(
        name_to_tags=name_to_tags,
        subtype_to_tags=subtype_to_tags,
        type_to_tags=type_to_tags,
    )
    logger.info(
        f"Loaded GOLDEN TAGS: name={len(name_tags)}, "
        f"subtype={len(subtype_tags)}, type={len(type_tags)}"
    )

    # Limit number of SKUs for testing
    if limit_skus is not None:
        sku_list = list(sku_data.keys())[:limit_skus]
        sku_data = {sku: sku_data[sku] for sku in sku_list}
        logger.info(f"Limited to {limit_skus} SKUs for processing")

    # Process each SKU
    results: List[Dict[str, Any]] = []
    total_skus = len(sku_data)
    processed = 0
    errors = 0

    for sku, product_data in sku_data.items():
        processed += 1
        logger.info(
            f"Processing SKU {processed}/{total_skus}: {sku} "
            f"({product_data['num_reviews']} reviews)"
        )

        try:
            # Run inference
            tags = run_inference(
                reviews=product_data["reviews"],
                llm_client=llm_client,
                name_to_tags=name_tags,
                subtype_to_tags=subtype_tags,
                type_to_tags=type_tags,
                product_name=product_data["name"],
                product_subtype=product_data["subtype"],
                product_type=product_data["type"],
                max_chars=max_chars,
                max_reviews=max_reviews,
                min_review_length=min_review_length,
                max_tags=max_tags,
                custom_prompt_template=custom_prompt_template,
            )

            results.append(
                {
                    "sku": sku,
                    "name": product_data["name"],
                    "subtype": product_data["subtype"],
                    "type": product_data["type"],
                    "tags": ", ".join(tags),
                    "num_tags": len(tags),
                    "num_reviews": product_data["num_reviews"],
                    "error": None,
                }
            )

            logger.info(f"✓ {sku}: received {len(tags)} tags")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"✗ Error for {sku}: {error_msg}")

            if skip_errors:
                results.append(
                    {
                        "sku": sku,
                        "name": product_data["name"],
                        "subtype": product_data["subtype"],
                        "type": product_data["type"],
                        "tags": "",
                        "num_tags": 0,
                        "num_reviews": product_data["num_reviews"],
                        "error": error_msg,
                    }
                )
                errors += 1
            else:
                raise

    # Create DataFrame with results
    df_results = pd.DataFrame(results)

    # Save results if path is specified
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Results saved to: {output_path}")

    # Output statistics
    logger.info(
        f"Processing completed: {processed} SKUs processed, "
        f"{errors} errors, {len(df_results[df_results['num_tags'] > 0])} SKUs with tags"
    )

    return df_results


def run_pipeline_for_sku(
    csv_path: str,
    sku: str,
    llm_client: "LLMClient",
    name_to_tags: Optional[Dict[str, List[str]]] = None,
    subtype_to_tags: Optional[Dict[str, List[str]]] = None,
    type_to_tags: Optional[Dict[str, List[str]]] = None,
    max_chars: int = 500,
    max_reviews: int = 50,
    min_review_length: int = 10,
    max_tags: int = 6,
    custom_prompt_template: Optional[str] = None,
) -> List[str]:
    """Run pipeline for one specific SKU.

    Args:
        csv_path: Path to CSV file with reviews.
        sku: Product SKU to process.
        llm_client: Client for working with LLM.
        name_to_tags: Dictionary mapping name -> list of tags.
        subtype_to_tags: Dictionary mapping subtype -> list of tags.
        type_to_tags: Dictionary mapping type -> list of tags.
        max_chars: Maximum review length in characters. Default: 500.
        max_reviews: Maximum number of reviews to process. Default: 50.
        min_review_length: Minimum review length to include. Default: 10.
        max_tags: Maximum number of tags to return. Default: 6.
        custom_prompt_template: Custom prompt template (optional).

    Returns:
        List of tags for given SKU.

    Raises:
        ValueError: If SKU not found in file.

    Examples:
        >>> class MockLLMClient:
        ...     def generate(self, prompt: str) -> str:
        ...         return "quality"
        >>> client = MockLLMClient()
        >>> tags = run_pipeline_for_sku(
        ...     "data/raw/lamoda_reviews.csv",
        ...     "MP002XW0O0SI",
        ...     client,
        ...     name_to_tags={"T-Shirt": ["quality"]}
        ... )
        >>> len(tags) >= 0
        True
    """
    logger.info(f"Processing single SKU: {sku}")
    
    # Load data
    sku_data = load_dataset(csv_path=csv_path, min_reviews_per_sku=1)

    if sku not in sku_data:
        available_skus = list(sku_data.keys())[:10]
        raise ValueError(
            f"SKU '{sku}' not found in file. "
            f"Available SKUs (first 10): {available_skus}"
        )

    product_data = sku_data[sku]

    # Load GOLDEN TAGS
    name_tags, subtype_tags, type_tags = load_golden_tags_from_dict(
        name_to_tags=name_to_tags,
        subtype_to_tags=subtype_to_tags,
        type_to_tags=type_to_tags,
    )

    # Run inference
    tags = run_inference(
        reviews=product_data["reviews"],
        llm_client=llm_client,
        name_to_tags=name_tags,
        subtype_to_tags=subtype_tags,
        type_to_tags=type_tags,
        product_name=product_data["name"],
        product_subtype=product_data["subtype"],
        product_type=product_data["type"],
        max_chars=max_chars,
        max_reviews=max_reviews,
        min_review_length=min_review_length,
        max_tags=max_tags,
        custom_prompt_template=custom_prompt_template,
    )

    logger.info(f"Processing completed: extracted {len(tags)} tags for SKU {sku}")
    return tags

