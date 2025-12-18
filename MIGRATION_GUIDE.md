# Migration Guide: Old → New Structure

This guide helps you migrate from the old codebase structure to the refactored one.

## 📊 Import Changes

### Old Imports → New Imports

#### Core Functions

```python
# OLD
from src.inference import run_inference
from src.pipeline import run_pipeline_for_file, run_pipeline_for_sku

# NEW
from src.core import run_inference, run_pipeline_for_file, run_pipeline_for_sku
# OR
from src import run_inference, run_pipeline_for_file, run_pipeline_for_sku
```

#### LLM Clients

```python
# OLD
from src.llm_client import LLMClient, BaseLLMClient
from src.openai_client import OpenAILLMClient

# NEW
from src.clients import LLMClient, OpenAIClient
# OR
from src import LLMClient, OpenAIClient
```

Note: `BaseLLMClient` has been removed - use `LLMClient` Protocol instead.

#### Data Loading

```python
# OLD
from src.data_loader import load_dataset, load_golden_tags_from_dict

# NEW
from src.utils import load_dataset, load_golden_tags_from_dict
# OR
from src import load_dataset, load_golden_tags_from_dict
```

#### Utility Functions

```python
# OLD
from src.preprocessing import prepare_reviews
from src.postprocessing import postprocess_tags
from src.prompt_builder import build_prompt, get_golden_tags_for_product
from src.llm_inference import run_llm

# NEW
from src.utils import (
    prepare_reviews,
    postprocess_tags,
    build_prompt,
    get_golden_tags_for_product,
    execute_llm,  # renamed from run_llm
)
# OR
from src import prepare_reviews, postprocess_tags, build_prompt, ...
```

## 🔧 Code Changes

### 1. Client Initialization

#### OLD
```python
from src.openai_client import OpenAILLMClient

# Hardcoded API key
client = OpenAILLMClient(
    api_key="sk-proj-...",  # ❌ Hardcoded
    model="gpt-4o-mini"
)
```

#### NEW
```python
from src import OpenAIClient, Config

# From environment variable (.env file)
client = OpenAIClient()  # ✅ Reads from .env

# Or with explicit configuration
client = OpenAIClient(
    api_key=Config.OPENAI_API_KEY,
    model=Config.OPENAI_MODEL,
    temperature=Config.OPENAI_TEMPERATURE,
    max_tokens=Config.OPENAI_MAX_TOKENS
)
```

### 2. Configuration Management

#### OLD
```python
# Settings scattered across code
MAX_REVIEWS = 50
MAX_TAGS = 6
# API key hardcoded in run script
API_KEY = "sk-proj-..."
```

#### NEW
```python
# Create .env file
"""
OPENAI_API_KEY=your_key_here
MAX_REVIEWS_PER_SKU=50
MAX_TAGS_PER_SKU=6
"""

# Use in code
from src import Config

max_reviews = Config.MAX_REVIEWS_PER_SKU
max_tags = Config.MAX_TAGS_PER_SKU
```

### 3. Mock Client for Testing

#### OLD
```python
# Duplicate MockLLMClient in example_usage.py and example_pipeline_usage.py
class MockLLMClient:
    def generate(self, prompt: str) -> str:
        return "tag1, tag2"
```

#### NEW
```python
# Single source in examples/mock_client.py
from examples.mock_client import MockLLMClient

client = MockLLMClient()
```

### 4. Function Renames

#### `run_llm` → `execute_llm`

```python
# OLD
from src.llm_inference import run_llm
response = run_llm(prompt, client)

# NEW
from src.utils import execute_llm
response = execute_llm(prompt, client)
```

#### `OpenAILLMClient` → `OpenAIClient`

```python
# OLD
from src.openai_client import OpenAILLMClient
client = OpenAILLMClient()

# NEW
from src.clients import OpenAIClient
client = OpenAIClient()
```

## 📁 File Structure Changes

### Moved Files

| Old Location | New Location |
|-------------|--------------|
| `src/llm_client.py` | `src/clients/llm_client.py` |
| `src/openai_client.py` | `src/clients/openai_client.py` |
| `src/inference.py` | `src/core/tag_inference.py` |
| `src/pipeline.py` | `src/core/pipeline.py` |
| `src/data_loader.py` | `src/utils/data.py` |
| `src/llm_inference.py` | `src/utils/llm_executor.py` |
| `src/preprocessing.py` | `src/utils/preprocessing.py` |
| `src/postprocessing.py` | `src/utils/postprocessing.py` |
| `src/prompt_builder.py` | `src/utils/prompt_builder.py` |
| `src/example_usage.py` | `examples/example_basic_inference.py` |
| `src/example_pipeline_usage.py` | `examples/example_pipeline.py` |

### New Files

- `src/config.py` - Configuration management
- `.env.example` - Environment variables template
- `examples/mock_client.py` - Shared mock client
- `MIGRATION_GUIDE.md` - This file

## 🚀 Quick Migration Steps

### Step 1: Update Dependencies

Ensure `python-dotenv` is installed:

```bash
poetry add python-dotenv
# OR
pip install python-dotenv
```

### Step 2: Create .env File

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Step 3: Update Imports in Your Code

Run find & replace in your IDE:

1. `from src.inference import` → `from src.core import`
2. `from src.pipeline import` → `from src.core import`
3. `from src.data_loader import` → `from src.utils import`
4. `from src.openai_client import OpenAILLMClient` → `from src.clients import OpenAIClient`
5. `from src.llm_client import` → `from src.clients import`
6. `run_llm(` → `execute_llm(`
7. `OpenAILLMClient(` → `OpenAIClient(`

### Step 4: Remove Hardcoded API Keys

Search for `sk-proj-` or `sk-` in your codebase and move to .env:

```python
# Before
API_KEY = "sk-proj-..."

# After
from src import Config
API_KEY = Config.OPENAI_API_KEY
```

### Step 5: Test Your Code

```bash
# Run examples to verify
python -m examples.example_basic_inference

# Run your main pipeline
python run_pipeline_openai.py --limit 5
```

## 🐛 Common Migration Issues

### Issue 1: ImportError

```python
ImportError: cannot import name 'OpenAILLMClient' from 'src.openai_client'
```

**Solution:** Update import to use new client name
```python
from src.clients import OpenAIClient  # Not OpenAILLMClient
```

### Issue 2: Missing .env File

```python
ValueError: API key not provided. Set OPENAI_API_KEY environment variable
```

**Solution:** Create .env file
```bash
cp .env.example .env
# Edit .env and add your API key
```

### Issue 3: BaseLLMClient Not Found

```python
ImportError: cannot import name 'BaseLLMClient'
```

**Solution:** Use `LLMClient` Protocol instead
```python
from src.clients import LLMClient  # Not BaseLLMClient
```

### Issue 4: run_llm Not Found

```python
ImportError: cannot import name 'run_llm'
```

**Solution:** Function was renamed
```python
from src.utils import execute_llm  # Not run_llm
```

## ✅ Migration Checklist

- [ ] Install `python-dotenv` dependency
- [ ] Create `.env` file from `.env.example`
- [ ] Add `OPENAI_API_KEY` to `.env`
- [ ] Update all imports in your code
- [ ] Remove hardcoded API keys
- [ ] Rename `run_llm` to `execute_llm`
- [ ] Rename `OpenAILLMClient` to `OpenAIClient`
- [ ] Test with mock client first
- [ ] Test with real API
- [ ] Update any custom scripts
- [ ] Update documentation/comments

## 📚 Additional Resources

- See `README.md` for complete documentation
- Check `examples/` folder for working examples
- Review `src/config.py` for all available configuration options

## 💡 Benefits After Migration

- ✅ No hardcoded secrets
- ✅ Better code organization
- ✅ Easier to test (mock client separated)
- ✅ Clearer naming conventions
- ✅ Production-ready configuration management
- ✅ Better logging and error handling
- ✅ Type safety with protocols

## 🤝 Need Help?

If you encounter issues during migration:

1. Check this guide's Common Issues section
2. Review the examples in `examples/` folder
3. Consult the main `README.md`
4. Check if `.env` file is properly configured

