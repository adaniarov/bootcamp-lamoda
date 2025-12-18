# Инструкция по использованию OpenAI LLM клиента

## Установка зависимостей

```bash
poetry add openai
# или
poetry install
```

## Использование

### Вариант 1: Запуск готового скрипта

```bash
python run_pipeline_openai.py
```

С параметрами:
```bash
python run_pipeline_openai.py --limit 10 --output results.csv
```

### Вариант 2: Использование в коде

```python
from src.openai_client import OpenAILLMClient
from src.pipeline import run_pipeline_for_file

# Создаем клиент
llm_client = OpenAILLMClient(
    api_key="your-api-key-here",  # или используйте переменную окружения
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=200,
)

# Запускаем pipeline
results = run_pipeline_for_file(
    csv_path="data/raw/lamoda_reviews.csv",
    llm_client=llm_client,
    name_to_tags={...},
    # ... остальные параметры
)
```

## Безопасность

⚠️ **ВАЖНО**: API ключ в файле `run_pipeline_openai.py` захардкожен для удобства, но это небезопасно!

Рекомендуется:
1. Использовать переменную окружения:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   python run_pipeline_openai.py
   ```

2. Или использовать файл `.env`:
   ```bash
   echo "OPENAI_API_KEY=your-api-key" > .env
   ```

3. Или передавать через параметр:
   ```bash
   python run_pipeline_openai.py --api-key "your-api-key"
   ```

## Параметры OpenAI клиента

- `api_key`: API ключ OpenAI (можно через переменную окружения)
- `model`: Модель (по умолчанию "gpt-4o-mini")
- `temperature`: Температура генерации 0.0-2.0 (по умолчанию 0.3)
- `max_tokens`: Максимальное количество токенов (по умолчанию 200)
- `store`: Сохранять ли запросы в истории OpenAI (по умолчанию True)

