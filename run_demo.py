"""Скрипт для запуска демонстрации LLM инференса."""

from pathlib import Path

from src.main import main

if __name__ == "__main__":
    # Запуск демонстрации с параметрами по умолчанию
    # Используется файл с одним примером из interim (создается через notebooks/get_one_example.ipynb)
    # Для обработки всех товаров используйте "data/raw/lamoda_reviews.csv" и установите limit=None
    
    example_file = Path("data/interim/one_example.csv")
    golden_tags_file = Path("data/processed/golden_tags.json")
    
    if not example_file.exists():
        print("⚠ Внимание: Файл data/interim/one_example.csv не найден!")
        print("Сначала запустите ноутбук notebooks/get_one_example.ipynb для создания файла с примером.")
        print("Или используйте полный файл: data/raw/lamoda_reviews.csv")
        exit(1)
    
    # Проверяем наличие файла с golden_tags (не критично, если его нет)
    if not golden_tags_file.exists():
        print(f"⚠ Внимание: Файл {golden_tags_file} не найден, будет использоваться без golden_tags")
        golden_tags_path = None
    else:
        golden_tags_path = str(golden_tags_file)
        print(f"✓ Используется файл с golden_tags: {golden_tags_file}")
    
    main(
        csv_path=str(example_file),  # Файл с одним случайным SKU
        golden_tags_path=golden_tags_path,  # Путь к файлу с GOLDEN_TAGS
        max_chars=500,
        max_reviews=10,
        min_length=10,
        limit=None,  # Для одного примера limit не нужен (или можно оставить 1)
        model="gpt-4o-mini",
    )

