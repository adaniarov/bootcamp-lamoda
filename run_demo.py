# run_demo.py
"""Скрипт для запуска демонстрации LLM / vectorize инференса."""

from pathlib import Path
import argparse

from src.main import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: Lamoda tags generation (LLM / vectorize)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["llm", "vectorize"],
        default="llm",
        help="Режим работы: 'llm' или 'vectorize' (по умолчанию: llm)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/one_example.csv",
        help="Путь к CSV с отзывами. "
             "Для LLM по умолчанию data/one_example.csv, "
             "для vectorize лучше указать полный lamoda_reviews.csv",
    )
    parser.add_argument(
        "--golden-tags-path",
        type=str,
        default="data/golden_tags_2.json",
        help="Путь к JSON с GOLDEN TAGS",
    )
    parser.add_argument(
        "--sku",
        type=str,
        default=None,
        help="SKU для обработки (обязательно в режиме vectorize)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=500,
        help="Максимальная длина одного отзыва в символах",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=10,
        help="Максимальное количество отзывов на SKU",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Минимальная длина отзыва для учёта",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Лимит SKU для LLM-режима (demо). В vectorize не используется.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Имя модели OpenAI для LLM-режима",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    csv_path = Path(args.csv_path)
    golden_tags_file = Path(args.golden_tags_path)

    # Проверка CSV
    if not csv_path.exists():
        print(f"⚠ Внимание: CSV файл {csv_path} не найден!")
        exit(1)

    # Проверка GOLDEN TAGS
    if not golden_tags_file.exists():
        print(f"⚠ Внимание: Файл {golden_tags_file} не найден.")
        golden_tags_path = None
    else:
        golden_tags_path = str(golden_tags_file)
        print(f"✓ Используется файл с golden_tags: {golden_tags_file}")

    # Особое правило для vectorize: нужен sku
    if args.mode == "vectorize" and not args.sku:
        print("⚠ Для режима 'vectorize' обязательно укажите --sku=<SKU>")
        exit(1)

    # Запуск
    main(
        csv_path=str(csv_path),
        golden_tags_path=golden_tags_path,
        max_chars=args.max_chars,
        max_reviews=args.max_reviews,
        min_length=args.min_length,
        limit=args.limit,          # используется только в LLM-режиме
        model=args.model,
        mode=args.mode,
        sku=args.sku,              # в vectorize обязателен, в llm игнорируется
    )