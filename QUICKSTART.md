# Quick Start Guide

Get up and running with the refactored Lamoda Review Tag Inference System in 5 minutes.

## ⚡ 5-Minute Setup

### 1. Install Dependencies

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install pandas openai python-dotenv
```

### 2. Configure Environment

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Open .env in your editor and set:
# OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Test with Mock Client (No API costs)

```bash
# Run example with mock client (instant, no API calls)
python -m examples.example_basic_inference
```

Expected output:
```
INFO:__main__:Starting inference: 4 reviews...
INFO:__main__:Inference completed: extracted 2 tags
INFO:__main__:Result: ['quality', 'size']
```

### 4. Run Real Pipeline (Uses OpenAI API)

```bash
# Test with 5 SKUs first (to avoid high costs)
python run_pipeline_openai.py --limit 5

# If successful, run on more data
python run_pipeline_openai.py --limit 100
```

## 📊 Quick Examples

### Example 1: Basic Tag Inference

```python
from src import OpenAIClient, run_inference

# Initialize
client = OpenAIClient()  # Reads key from .env

# Your data
reviews = ["Great quality!", "Perfect size", "Good value"]
golden_tags = {"T-Shirt": ["quality", "size", "price", "color"]}

# Run inference
tags = run_inference(
    reviews=reviews,
    llm_client=client,
    name_to_tags=golden_tags,
    subtype_to_tags={},
    type_to_tags={},
    product_name="T-Shirt"
)

print(f"Tags: {tags}")  # ['quality', 'size', 'price']
```

### Example 2: Batch Processing

```python
from src import OpenAIClient, run_pipeline_for_file

# Initialize
client = OpenAIClient()

# Your golden tags
golden_tags = {
    "T-Shirt": ["quality", "size", "material"],
    "Jeans": ["quality", "size", "fit"],
}

# Process file
results = run_pipeline_for_file(
    csv_path="data/raw/reviews.csv",
    llm_client=client,
    name_to_tags=golden_tags,
    output_path="data/processed/results.csv",
    limit_skus=10  # Start small!
)

print(f"Processed {len(results)} SKUs")
```

### Example 3: Using Mock Client (Testing)

```python
from examples.mock_client import MockLLMClient
from src import run_inference

# No API costs!
client = MockLLMClient()

tags = run_inference(
    reviews=["Test review"],
    llm_client=client,
    name_to_tags={"Product": ["quality"]},
    subtype_to_tags={},
    type_to_tags={},
    product_name="Product"
)
```

## 🔧 Common Commands

### Run with Different Settings

```bash
# Use different model
python run_pipeline_openai.py --model gpt-4 --limit 10

# Change output location
python run_pipeline_openai.py --output my_results.csv

# Debug mode
python run_pipeline_openai.py --log-level DEBUG --limit 5

# Custom input file
python run_pipeline_openai.py --input my_data.csv --limit 10
```

### Environment Variables

Edit `.env` file to change defaults:

```env
# API Configuration
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.3
OPENAI_MAX_TOKENS=200

# Processing Limits
MAX_REVIEWS_PER_SKU=50
MAX_TAGS_PER_SKU=6
MAX_CHARS_PER_REVIEW=500

# Paths
INPUT_CSV_PATH=data/raw/lamoda_reviews.csv
OUTPUT_CSV_PATH=data/processed/results.csv
```

## 📁 Required CSV Format

Your input CSV should have:

| Column | Required | Description |
|--------|----------|-------------|
| `product_sku` | ✓ | Unique product identifier |
| `comment_text` | ✓ | Review text |
| `name` | Optional | Product name |
| `good_subtype` | Optional | Product subtype |
| `good_type` | Optional | Product type |

Example:
```csv
product_sku,comment_text,name,good_subtype,good_type
MP001,Great quality!,T-Shirt,TEE-SHIRTS,Clothes
MP001,Perfect size,T-Shirt,TEE-SHIRTS,Clothes
MP002,Love these jeans,Jeans,JEANS,Clothes
```

## 🎯 Golden Tags Setup

Golden tags are the valid tags for each product. Set them up in `run_pipeline_openai.py`:

```python
def load_golden_tags():
    name_to_tags = {
        "T-Shirt": ["quality", "size", "material", "price"],
        "Jeans": ["quality", "size", "fit", "price"],
        # Add your products here
    }
    
    subtype_to_tags = {
        "TEE-SHIRTS": ["quality", "size", "material"],
        "JEANS": ["quality", "size", "fit"],
        # Add your subtypes here
    }
    
    type_to_tags = {
        "Clothes": ["quality", "size", "price"],
        "Shoes": ["quality", "size", "comfort"],
        # Add your types here
    }
    
    return name_to_tags, subtype_to_tags, type_to_tags
```

## ✅ Verification

Test your setup:

```bash
# 1. Test imports work
python -c "from src import Config, OpenAIClient; print('✓ Imports OK')"

# 2. Test mock client
python -m examples.example_basic_inference

# 3. Test with real API (small batch)
python run_pipeline_openai.py --limit 5
```

## 🐛 Troubleshooting

### Error: "No module named 'openai'"

```bash
poetry add openai
# or
pip install openai
```

### Error: "No module named 'dotenv'"

```bash
poetry add python-dotenv
# or
pip install python-dotenv
```

### Error: "API key not provided"

1. Check `.env` file exists
2. Check `OPENAI_API_KEY` is set in `.env`
3. No spaces around `=` in `.env`

### Error: "File not found: data/raw/lamoda_reviews.csv"

Either:
- Place your CSV at that location, or
- Use `--input` flag: `python run_pipeline_openai.py --input your_file.csv`

## 📚 Next Steps

1. **Read the docs:**
   - `README.md` - Full documentation
   - `MIGRATION_GUIDE.md` - If migrating from old code
   - `REFACTORING_SUMMARY.md` - What changed

2. **Explore examples:**
   - `examples/example_basic_inference.py`
   - `examples/example_pipeline.py`

3. **Customize for your needs:**
   - Edit golden tags in `run_pipeline_openai.py`
   - Adjust settings in `.env`
   - Add your own LLM providers in `src/clients/`

## 💡 Tips

- **Start small:** Always use `--limit` flag when testing
- **Use mock client:** Test logic without API costs
- **Monitor costs:** OpenAI charges per token
- **Check logs:** Use `--log-level DEBUG` to see details
- **Backup data:** Always keep original CSV files

## 🆘 Need Help?

1. Check `TROUBLESHOOTING.md` (if exists)
2. Review error messages carefully
3. Verify `.env` configuration
4. Test with mock client first
5. Start with `--limit 1` to debug

---

**Ready to go!** 🚀

Start with:
```bash
python run_pipeline_openai.py --limit 5
```

