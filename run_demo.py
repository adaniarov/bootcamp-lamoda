"""Скрипт для запуска демонстрации LLM инференса."""

from src.main import main

if __name__ == "__main__":
    # Запуск демонстрации с параметрами по умолчанию
    # Для обработки всех товаров установите limit=None
    main(
        csv_path="data/raw/lamoda_reviews.csv",
        golden_tags_path=None,  # Укажите путь к файлу с GOLDEN_TAGS, если есть
        max_chars=500,
        max_reviews=10,
        min_length=10,
        limit=5,  # Для демонстрации обрабатываем только 5 товаров
        model="gpt-4o-mini",
    )

