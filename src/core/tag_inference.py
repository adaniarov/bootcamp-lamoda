"""Main module for executing tag inference using LLM."""

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

from src.utils import (
    build_prompt,
    execute_llm,
    get_golden_tags_for_product,
    postprocess_tags,
    prepare_reviews,
)

if TYPE_CHECKING:
    from src.clients import LLMClient

logger = logging.getLogger(__name__)


def run_inference(
    reviews: List[str],
    llm_client: "LLMClient",
    name_to_tags: Dict[str, List[str]],
    subtype_to_tags: Dict[str, List[str]],
    type_to_tags: Dict[str, List[str]],
    product_name: Optional[str] = None,
    product_subtype: Optional[str] = None,
    product_type: Optional[str] = None,
    max_chars: int = 500,
    max_reviews: int = 50,
    min_review_length: int = 10,
    max_tags: int = 6,
    custom_prompt_template: Optional[str] = None,
) -> List[str]:
    """Execute full inference cycle: preprocessing -> prompt -> LLM -> postprocessing.

    Args:
        reviews: List of original reviews for SKU.
        llm_client: Client for working with LLM.
        name_to_tags: Dictionary mapping name -> list of tags.
        subtype_to_tags: Dictionary mapping subtype -> list of tags.
        type_to_tags: Dictionary mapping type -> list of tags.
        product_name: Product name (optional).
        product_subtype: Product subtype (optional).
        product_type: Product type (optional).
        max_chars: Maximum review length in characters. Default: 500.
        max_reviews: Maximum number of reviews to process. Default: 50.
        min_review_length: Minimum review length to include. Default: 10.
        max_tags: Maximum number of tags to return. Default: 6.
        custom_prompt_template: Custom prompt template (optional).

    Returns:
        List of valid tags (maximum max_tags).

    Raises:
        ValueError: If reviews is empty or couldn't get GOLDEN TAGS.
        Exception: On LLM call error.

    Examples:
        >>> class MockLLMClient:
        ...     def generate(self, prompt: str) -> str:
        ...         return "quality, size"
        >>> client = MockLLMClient()
        >>> reviews = ["Excellent quality", "Good size"]
        >>> name_to_tags = {"T-Shirt": ["quality", "size", "price"]}
        >>> result = run_inference(
        ...     reviews, client, name_to_tags, {}, {},
        ...     product_name="T-Shirt"
        ... )
        >>> len(result) > 0
        True
    """
    logger.info(
        f"Starting inference: {len(reviews)} reviews, "
        f"product={product_name or product_subtype or product_type}"
    )

    # 1. Preprocess reviews
    processed_reviews = prepare_reviews(
        reviews=reviews,
        max_chars=max_chars,
        max_reviews=max_reviews,
        min_review_length=min_review_length,
    )

    if not processed_reviews:
        logger.warning("No valid reviews after preprocessing")
        return []

    # 2. Get GOLDEN TAGS for product
    golden_tags = get_golden_tags_for_product(
        product_name=product_name,
        product_subtype=product_subtype,
        product_type=product_type,
        name_to_tags=name_to_tags,
        subtype_to_tags=subtype_to_tags,
        type_to_tags=type_to_tags,
    )

    if not golden_tags:
        logger.warning("No golden tags found for product")
        return []

    # 3. Build prompt
    prompt = build_prompt(
        reviews=processed_reviews,
        golden_tags=golden_tags,
        product_name=product_name,
        product_subtype=product_subtype,
        product_type=product_type,
        custom_prompt_template=custom_prompt_template,
    )

    # 4. Call LLM
    llm_response = execute_llm(prompt=prompt, llm_client=llm_client)

    # 5. Postprocess tags
    tags = postprocess_tags(
        llm_response=llm_response,
        golden_tags=golden_tags,
        max_tags=max_tags,
    )

    logger.info(f"Inference completed: extracted {len(tags)} tags")
    return tags

