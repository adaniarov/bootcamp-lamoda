<<<<<<< Current (Your changes)
# Lamoda Bootcamp

Проект для анализа отзывов Lamoda с использованием LLM.
=======
# Lamoda Review Tag Inference System

Production-ready LLM-based system for analyzing product reviews and extracting relevant tags using OpenAI API.

## 🏗️ Project Structure

```
oez/
├── src/                          # Main source code
│   ├── clients/                  # LLM client implementations
│   │   ├── llm_client.py        # Protocol interface for LLM clients
│   │   └── openai_client.py     # OpenAI API client implementation
│   ├── core/                     # Core business logic
│   │   ├── tag_inference.py     # Main inference pipeline
│   │   └── pipeline.py          # Batch processing pipelines
│   ├── utils/                    # Utility functions
│   │   ├── data.py              # Data loading utilities
│   │   ├── preprocessing.py     # Review preprocessing
│   │   ├── postprocessing.py    # Tag postprocessing
│   │   ├── prompt_builder.py    # LLM prompt construction
│   │   └── llm_executor.py      # LLM execution with retries
│   └── config.py                 # Configuration management
├── examples/                     # Usage examples
│   ├── mock_client.py           # Mock LLM client for testing
│   ├── example_basic_inference.py
│   └── example_pipeline.py
├── notebooks/                    # Jupyter notebooks for experiments
├── data/                         # Data directory
│   ├── raw/                     # Raw data files
│   └── processed/               # Processed results
├── .env.example                  # Environment variables template
├── run_pipeline_openai.py       # Main script for running pipeline
└── pyproject.toml               # Project dependencies

```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd oez

# Install dependencies using Poetry
poetry install

# Or using pip
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` and set your OpenAI API key:

```env
OPENAI_API_KEY=your_actual_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.3
OPENAI_MAX_TOKENS=200

# Pipeline Configuration
MAX_REVIEWS_PER_SKU=50
MAX_TAGS_PER_SKU=6
MAX_CHARS_PER_REVIEW=500
MIN_REVIEW_LENGTH=10
MIN_REVIEWS_PER_SKU=1

# Data Paths
INPUT_CSV_PATH=data/raw/lamoda_reviews.csv
OUTPUT_CSV_PATH=data/processed/llm_tags_results.csv

# Logging
LOG_LEVEL=INFO
```

### 3. Running the Pipeline

#### Basic Usage

```bash
# Run with default settings from .env
python run_pipeline_openai.py
```

#### Advanced Usage

```bash
# Process limited number of SKUs for testing
python run_pipeline_openai.py --limit 10

# Override input/output paths
python run_pipeline_openai.py --input data.csv --output results.csv

# Use different OpenAI model
python run_pipeline_openai.py --model gpt-4

# Adjust verbosity
python run_pipeline_openai.py --log-level DEBUG
```

## 📚 Usage Examples

### Basic Inference

```python
from src import OpenAIClient, run_inference

# Initialize client
client = OpenAIClient()  # Reads API key from .env

# Prepare data
reviews = ["Great quality!", "Perfect size"]
name_to_tags = {"T-Shirt": ["quality", "size", "price"]}

# Run inference
tags = run_inference(
    reviews=reviews,
    llm_client=client,
    name_to_tags=name_to_tags,
    subtype_to_tags={},
    type_to_tags={},
    product_name="T-Shirt"
)

print(f"Extracted tags: {tags}")
```

### Batch Processing

```python
from src import OpenAIClient, run_pipeline_for_file

# Initialize client
client = OpenAIClient()

# Prepare golden tags
golden_tags = {
    "T-Shirt": ["quality", "size", "material"],
    "Jeans": ["quality", "size", "fit"]
}

# Run pipeline for entire file
results = run_pipeline_for_file(
    csv_path="data/raw/reviews.csv",
    llm_client=client,
    name_to_tags=golden_tags,
    output_path="data/processed/results.csv",
    limit_skus=100  # Optional: limit for testing
)

print(f"Processed {len(results)} SKUs")
```

### Using Mock Client for Testing

```python
from examples.mock_client import MockLLMClient
from src import run_inference

# No API costs, instant responses
client = MockLLMClient()

tags = run_inference(
    reviews=["Great product"],
    llm_client=client,
    name_to_tags={"Product": ["quality", "price"]},
    subtype_to_tags={},
    type_to_tags={},
    product_name="Product"
)
```

## 🏭 Production Features

### Security & Configuration
- ✅ **Environment Variables**: All sensitive data (API keys) in `.env` file
- ✅ **Configuration Management**: Centralized config in `src/config.py`
- ✅ **Type Safety**: Full type hints for better IDE support and error detection

### Code Quality
- ✅ **Clean Architecture**: Separated concerns (clients, core, utils)
- ✅ **Protocol-based Design**: Easy to swap LLM providers
- ✅ **Logging**: Comprehensive logging throughout the codebase
- ✅ **Error Handling**: Retry logic and graceful error handling

### Maintainability
- ✅ **No Code Duplication**: Single source of truth for each functionality
- ✅ **Clear Naming**: Self-documenting code with descriptive names
- ✅ **Modular Design**: Easy to extend and test
- ✅ **Examples Separated**: Production code separate from examples

## 🔧 Key Improvements from Original Code

### 1. **Removed Duplications**
- Merged `BaseLLMClient` and `LLMClient` into single Protocol
- Consolidated `MockLLMClient` into examples package
- Eliminated redundant functionality across modules

### 2. **Clearer File Names**
- `inference.py` → `tag_inference.py` (more specific)
- `llm_inference.py` → `llm_executor.py` (clearer purpose)
- `data_loader.py` → `data.py` (cleaner name)
- `openai_client.py` → `clients/openai_client.py` (better organization)

### 3. **Production-Ready Features**
- Environment variable management with `python-dotenv`
- Centralized configuration in `Config` class
- Comprehensive logging at all levels
- Proper error handling and retries
- Type hints throughout

### 4. **Security**
- No hardcoded API keys
- API keys loaded from environment variables
- `.env.example` template for easy setup
- `.env` in `.gitignore` (if not already)

## 📖 API Reference

### Core Functions

#### `run_inference()`
Execute full inference cycle for a single product.

**Parameters:**
- `reviews`: List of review strings
- `llm_client`: LLM client implementing `LLMClient` protocol
- `name_to_tags`, `subtype_to_tags`, `type_to_tags`: Golden tags dictionaries
- `product_name`, `product_subtype`, `product_type`: Product metadata
- `max_chars`, `max_reviews`, `min_review_length`, `max_tags`: Processing parameters

**Returns:** List of extracted tags

#### `run_pipeline_for_file()`
Process entire CSV file with reviews.

**Parameters:**
- `csv_path`: Path to input CSV
- `llm_client`: LLM client
- Golden tags dictionaries
- `output_path`: Where to save results
- Processing parameters
- `limit_skus`: Optional limit for testing
- `skip_errors`: Continue on errors

**Returns:** pandas DataFrame with results

### Clients

#### `OpenAIClient`
Production client for OpenAI API.

```python
client = OpenAIClient(
    api_key="...",  # Optional, reads from OPENAI_API_KEY env var
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=200
)
```

#### `MockLLMClient`
Test client that doesn't make real API calls.

```python
from examples.mock_client import MockLLMClient
client = MockLLMClient()
```

## 🧪 Testing

Run the examples to verify your setup:

```bash
# Basic inference example
python -m examples.example_basic_inference

# Pipeline example (requires data file)
python -m examples.example_pipeline
```

## 📊 Expected Input Format

The CSV file should have these columns:
- `product_sku`: Unique product identifier
- `comment_text`: Review text
- `name`: Product name (optional)
- `good_subtype`: Product subtype (optional)
- `good_type`: Product type (optional)

## 📝 TODO

- [ ] Add support for other LLM providers (Anthropic, Cohere, etc.)
- [ ] Add caching layer for LLM responses
- [ ] Add metrics and monitoring
- [ ] Add unit tests
- [ ] Add CI/CD pipeline

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Ensure code follows the existing style
4. Update documentation if needed
5. Submit a pull request

## 📄 License

[Your License Here]

## 🔗 Links

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Python-dotenv Documentation](https://github.com/theskumar/python-dotenv)
>>>>>>> Incoming (Background Agent changes)
