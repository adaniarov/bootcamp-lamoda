"""Пример использования модулей для LLM инференса отзывов."""

from pathlib import Path

from src.data_loader import load_dataset, load_golden_tags
from src.openai_client import OpenAILLMClient
from src.llm_inference import run_llm
from src.pipeline import run_pipeline_for_file, run_pipeline_for_sku
from src.preprocessing import prepare_reviews

# Пример 1: Загрузка данных
print("Пример 1: Загрузка данных")
df = load_dataset("data/raw/lamoda_reviews.csv")
print(f"Загружено {len(df)} отзывов")
print(f"Уникальных товаров: {df['product_sku'].nunique()}\n")

# Пример 2: Предобработка отзывов
print("Пример 2: Предобработка отзывов")
reviews = df[df["product_sku"] == df["product_sku"].iloc[0]]["comment_text"].dropna().tolist()
processed_reviews = prepare_reviews(reviews, max_chars=500, max_reviews=10, min_length=10)
print(f"Исходных отзывов: {len(reviews)}")
print(f"После обработки: {len(processed_reviews)}\n")

# Пример 3: Инициализация LLM клиента
print("Пример 3: Инициализация LLM клиента")
# Убедитесь, что создан .env файл с OPENAI_API_KEY
try:
    llm_client = OpenAILLMClient(model="gpt-4o-mini")
    print("✓ LLM клиент инициализирован\n")
except ValueError as e:
    print(f"✗ Ошибка: {e}")
    print("Создайте файл .env с переменной OPENAI_API_KEY\n")

# Пример 4: Загрузка GOLDEN_TAGS (если есть)
print("Пример 4: Загрузка GOLDEN_TAGS")
golden_tags_path = Path("data/external/golden_tags.json")
if golden_tags_path.exists():
    golden_tags = load_golden_tags(golden_tags_path)
    print(f"Загружено {len(golden_tags)} golden tags\n")
else:
    print("Файл с GOLDEN_TAGS не найден (это нормально, если не используется)\n")
    golden_tags = None

# Пример 5: Вызов LLM для одного товара
print("Пример 5: Вызов LLM для одного товара")
if "llm_client" in locals():
    first_sku = df["product_sku"].iloc[0]
    first_name = df[df["product_sku"] == first_sku]["name"].iloc[0]
    first_reviews = df[df["product_sku"] == first_sku]["comment_text"].dropna().tolist()
    
    try:
        result = run_llm(
            sku=first_sku,
            product_name=first_name,
            reviews=first_reviews,
            llm_client=llm_client,
            golden_tags=golden_tags,
            max_chars=500,
            max_reviews=10,
            min_length=10,
        )
        print(f"Результат: {result}\n")
    except Exception as e:
        print(f"Ошибка при вызове LLM: {e}\n")

# Пример 6: Запуск для всего датасета (ограничено 3 товарами для примера)
print("Пример 6: Запуск для всего датасета (ограничено 3 товарами)")
if "llm_client" in locals():
    try:
        results = run_pipeline_for_file(
            csv_path="data/raw/lamoda_reviews.csv",
            llm_client=llm_client,
            golden_tags_path=golden_tags_path if golden_tags_path.exists() else None,
            max_chars=500,
            max_reviews=10,
            min_length=10,
            limit=3,  # Ограничение для примера
        )
        print(f"Обработано товаров: {len(results)}")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result}")
    except Exception as e:
        print(f"Ошибка: {e}")

