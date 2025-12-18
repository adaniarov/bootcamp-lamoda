"""Скрипт для запуска демонстрации LLM инференса."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

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
    
    # Запускаем обработку
    results_df = main(
        csv_path=str(example_file),  # Файл с одним случайным SKU
        golden_tags_path=golden_tags_path,  # Путь к файлу с GOLDEN_TAGS
        max_chars=500,
        max_reviews=10,
        min_length=10,
        limit=None,  # Для одного примера limit не нужен (или можно оставить 1)
        model="gpt-4o-mini",
    )

    # Сохраняем результаты в JSON файл
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Формируем имя файла с timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.json"

    # Преобразуем DataFrame в список словарей для JSON
    results_list = []
    for _, row in results_df.iterrows():
        # Обрабатываем теги: если строка, разбиваем по запятым, если уже список - используем как есть
        tags_value = row.get("tags", "")
        if isinstance(tags_value, str) and tags_value:
            tags_list = [tag.strip() for tag in tags_value.split(",") if tag.strip()]
        elif isinstance(tags_value, list):
            tags_list = tags_value
        else:
            tags_list = []

        result_dict = {
            "sku": str(row["sku"]),
            "name": str(row.get("name", "")),
            "subtype": str(row.get("subtype", "")) if pd.notna(row.get("subtype")) else None,
            "type": str(row.get("type", "")) if pd.notna(row.get("type")) else None,
            "tags": tags_list,
            "num_tags": int(row.get("num_tags", 0)),
            "num_reviews": int(row.get("num_reviews", 0)),
        }
        if row.get("error") and pd.notna(row.get("error")):
            result_dict["error"] = str(row["error"])
        results_list.append(result_dict)

    # Сохраняем в JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "input_file": str(example_file),
        "golden_tags_file": golden_tags_path,
        "total_processed": len(results_df),
        "with_tags": int(len(results_df[results_df["num_tags"] > 0])),
        "without_tags": int(len(results_df[results_df["num_tags"] == 0])),
        "results": results_list,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Результаты сохранены в JSON файл: {output_file}")
    print("=" * 80)

