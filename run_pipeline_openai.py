#!/usr/bin/env python3
"""Script for running pipeline with OpenAI LLM client.

This script demonstrates how to use the refactored codebase with proper
configuration management through environment variables.

Usage:
    # Using .env file (recommended):
    python run_pipeline_openai.py

    # Or with command-line parameters:
    python run_pipeline_openai.py --limit 10 --output results.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

from src import Config, OpenAIClient, run_pipeline_for_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# GOLDEN TAGS CONFIGURATION
# ============================================================================
# TODO: Replace with your actual golden tags data
# Consider loading this from a JSON/CSV file for better maintainability
def load_golden_tags() -> tuple[
    Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]
]:
    """Load GOLDEN TAGS dictionaries.

    IMPORTANT: Replace with your real data!
    Consider loading from JSON/CSV file for production use.
    
    Returns:
        Tuple of (name_to_tags, subtype_to_tags, type_to_tags) dictionaries.
    """
    # Example data - replace with yours!
    name_to_tags: Dict[str, List[str]] = {
        "Футболка": ["качество", "размер", "материал", "цена", "цвет"],
        "Джинсы": ["качество", "размер", "посадка", "цена"],
        "Носки": ["качество", "размер", "материал"],
        "Куртка утепленная": ["качество", "размер", "тепло", "цена"],
        "Свитшот": ["качество", "размер", "материал", "цена"],
    }

    subtype_to_tags: Dict[str, List[str]] = {
        "TEE-SHIRTS & POLOS": ["качество", "размер", "материал"],
        "JEANS": ["качество", "размер", "посадка"],
        "SOCKS & TIGHTS": ["качество", "размер"],
        "OUTWEAR": ["качество", "размер", "тепло"],
        "SWEATSHIRTS": ["качество", "размер", "материал"],
    }

    type_to_tags: Dict[str, List[str]] = {
        "Clothes": ["качество", "размер", "цена"],
        "Shoes": ["качество", "размер", "комфорт"],
        "Beauty_Accs": ["качество", "эффект", "цена"],
    }

    return name_to_tags, subtype_to_tags, type_to_tags


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main function for running pipeline with OpenAI."""
    parser = argparse.ArgumentParser(
        description="Run full Lamoda review processing pipeline with OpenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings from .env file:
  python run_pipeline_openai.py

  # Process limited number of SKUs for testing:
  python run_pipeline_openai.py --limit 10

  # Override input/output paths:
  python run_pipeline_openai.py --input data.csv --output results.csv

  # Use different OpenAI model:
  python run_pipeline_openai.py --model gpt-4

Note: 
  Set OPENAI_API_KEY in .env file or environment variables.
  See .env.example for all available configuration options.
        """,
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=f"Path to input CSV file (default from .env: {Config.INPUT_CSV_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Path to save results (default from .env: {Config.OUTPUT_CSV_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of SKUs to process (for testing)",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=None,
        help=f"Maximum reviews per SKU (default from .env: {Config.MAX_REVIEWS_PER_SKU})",
    )
    parser.add_argument(
        "--max-tags",
        type=int,
        default=None,
        help=f"Maximum tags per SKU (default from .env: {Config.MAX_TAGS_PER_SKU})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"OpenAI model (default from .env: {Config.OPENAI_MODEL})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help=f"Logging level (default from .env: {Config.LOG_LEVEL})",
    )

    args = parser.parse_args()

    # Set log level
    log_level = args.log_level or Config.LOG_LEVEL
    logging.getLogger().setLevel(getattr(logging, log_level))

    # Get configuration values (CLI args override .env)
    input_path = Path(args.input or Config.INPUT_CSV_PATH)
    output_path = args.output or Config.OUTPUT_CSV_PATH
    max_reviews = args.max_reviews or Config.MAX_REVIEWS_PER_SKU
    max_tags = args.max_tags or Config.MAX_TAGS_PER_SKU
    model = args.model or Config.OPENAI_MODEL

    # Check input file exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info(
            "Please ensure the file exists or set INPUT_CSV_PATH in .env file"
        )
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("LAMODA REVIEW PROCESSING PIPELINE WITH OPENAI")
    logger.info("=" * 80)
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Model: {model}")
    if args.limit:
        logger.info(f"Limit: {args.limit} SKUs")
    logger.info("=" * 80)

    # 1. Initialize OpenAI client
    logger.info("\n[1/3] Initializing OpenAI client...")
    try:
        # Validate configuration
        Config.validate()
        
        # Create client using configuration
        llm_client = OpenAIClient(
            api_key=Config.OPENAI_API_KEY,
            model=model,
            temperature=Config.OPENAI_TEMPERATURE,
            max_tokens=Config.OPENAI_MAX_TOKENS,
            store=True,
        )
        logger.info("✓ OpenAI client ready")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.info(
            "Please set OPENAI_API_KEY in .env file or environment variables. "
            "See .env.example for reference."
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {e}")
        sys.exit(1)

    # 2. Load GOLDEN TAGS
    logger.info("\n[2/3] Loading GOLDEN TAGS...")
    name_to_tags, subtype_to_tags, type_to_tags = load_golden_tags()
    logger.info(f"✓ Loaded:")
    logger.info(f"  - name_to_tags: {len(name_to_tags)} entries")
    logger.info(f"  - subtype_to_tags: {len(subtype_to_tags)} entries")
    logger.info(f"  - type_to_tags: {len(type_to_tags)} entries")

    # 3. Run pipeline
    logger.info("\n[3/3] Running processing pipeline...")
    logger.info("-" * 80)

    try:
        results_df = run_pipeline_for_file(
            csv_path=str(input_path),
            llm_client=llm_client,
            name_to_tags=name_to_tags,
            subtype_to_tags=subtype_to_tags,
            type_to_tags=type_to_tags,
            output_path=output_path,
            max_chars=Config.MAX_CHARS_PER_REVIEW,
            max_reviews=max_reviews,
            min_review_length=Config.MIN_REVIEW_LENGTH,
            max_tags=max_tags,
            min_reviews_per_sku=Config.MIN_REVIEWS_PER_SKU,
            limit_skus=args.limit,
            skip_errors=True,
        )

        # Output statistics
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total SKUs processed: {len(results_df)}")
        logger.info(f"SKUs with tags: {len(results_df[results_df['num_tags'] > 0])}")
        logger.info(f"SKUs without tags: {len(results_df[results_df['num_tags'] == 0])}")
        
        if 'error' in results_df.columns:
            logger.info(f"Errors: {len(results_df[results_df['error'].notna()])}")

        # Show example results
        logger.info("\nExample results:")
        logger.info("-" * 80)
        successful = results_df[results_df["num_tags"] > 0].head(5)
        for _, row in successful.iterrows():
            logger.info(f"SKU: {row['sku']}")
            logger.info(f"  Name: {row['name']}")
            logger.info(f"  Tags ({row['num_tags']}): {row['tags']}")
            logger.info("")

        logger.info(f"\n✓ Results saved to: {output_path}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\nError running pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
