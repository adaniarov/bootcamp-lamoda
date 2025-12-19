# src/main.py
"""Главный модуль для запуска демонстрации."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .openai_client import OpenAILLMClient
from .pipeline import run_llm_pipeline_for_file, run_pipeline_for_sku
from .vector_pipeline import (
    run_vector_pipeline_for_file,
    run_vector_pipeline_for_sku,
)


def _normalize_path(path: str | Path) -> Path:
    """Делает путь абсолютным относительно корня проекта (src/..)."""
    p = Path(path)
    if p.is_absolute():
        return p
    project_root = Path(__file__).parent.parent
    return project_root / p


def _load_golden_tags_name_dict(golden_tags_path: Path) -> Dict[str, List[str]]:
    """
    Упрощённый лоадер для LLM-ветки:
    превращает JSON в name_to_tags.
    """
    with open(golden_tags_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    name_to_tags: Dict[str, List[str]] = {}

    for item in data:
        if "name" not in item or "tags" not in item:
            continue

        name = item["name"]
        tags = item["tags"]
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        if not isinstance(tags, list):
            continue

        if name not in name_to_tags:
            name_to_tags[name] = []
        name_to_tags[name].extend(tags)

    # убираем дубли
    name_to_tags = {k: sorted(set(v)) for k, v in name_to_tags.items()}
    return name_to_tags


def main(
    csv_path: str | Path = "data/raw/lamoda_reviews.csv",
    golden_tags_path: str | Path | None = None,
    max_chars: int = 500,
    max_reviews: int = 10,
    min_length: int = 10,
    limit: int | None = 5,
    model: str = "gpt-4o-mini",
    mode: str = "llm",  # "llm" или "vectorize"
    sku: str | None = None,
):
    """
    Точка входа демо:

    - mode="llm":   обрабатываем много SKU из csv, печатаем DataFrame.
    - mode="vectorize": работаем ТОЛЬКО с одной sku, берём отзывы из csv и
      возвращаем для неё теги.
    """
    csv_path = _normalize_path(csv_path)

    if golden_tags_path:
        golden_tags_path = _normalize_path(golden_tags_path)

    print("=" * 80)
    print(f"ЗАПУСК ДЕМОНСТРАЦИИ LAMODA В РЕЖИМЕ: {mode}")
    print("=" * 80)
    print(f"CSV файл: {csv_path}")
    print(f"GOLDEN_TAGS файл: {golden_tags_path or 'не указан'}")
    print(f"Параметры: max_chars={max_chars}, max_reviews={max_reviews}, min_length={min_length}")
    print(f"Обработка товаров (limit для LLM): {limit}")
    print("=" * 80)

    # ---------------- LLM режим: много SKU ----------------
    if mode == "llm":
        print("\nИнициализация OpenAI клиента...")
        llm_client = OpenAILLMClient(model=model)
        print("✓ LLM клиент инициализирован")

        name_to_tags = None

        if golden_tags_path:
            print(f"\nЗагрузка GOLDEN_TAGS из {golden_tags_path}...")
            try:
                name_to_tags = _load_golden_tags_name_dict(golden_tags_path)
                print(f"✓ Загружено {len(name_to_tags)} записей golden_tags (по name)")
            except Exception as e:
                print(f"⚠ Ошибка при загрузке golden_tags: {e}")
                print("Продолжаем без golden_tags")

        print("\nЗапуск LLM-пайплайна по файлу...")
        results_df = run_llm_pipeline_for_file(
            csv_path=str(csv_path),
            llm_client=llm_client,
            name_to_tags=name_to_tags,
            subtype_to_tags=None,
            type_to_tags=None,
            max_chars=max_chars,
            max_reviews=max_reviews,
            min_review_length=min_length,
            max_tags=6,
            min_reviews_per_sku=1,
            custom_prompt_template=None,
            skip_errors=True,
            limit_skus=limit,
        )

        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТЫ LLM:")
        print("=" * 80)
        for i, row in results_df.iterrows():
            print(f"{i+1}. SKU: {row['sku']}")
            print(f"   Название: {row['name']}")
            print(f"   Теги ({row['num_tags']}): {row['tags'] if row['tags'] else 'нет тегов'}")
            if row.get("error"):
                print(f"   ⚠ Ошибка: {row['error']}")
            print()

        print("=" * 80)
        print(f"Обработано товаров: {len(results_df)}")
        print(f"С тегами: {len(results_df[results_df['num_tags'] > 0])}")
        print(f"Без тегов: {len(results_df[results_df['num_tags'] == 0])}")
        print("=" * 80)

        return results_df

    # ---------------- VECTORIZE режим: одна SKU ----------------
    elif mode == "vectorize":
        if not golden_tags_path:
            raise ValueError("Для режима 'vectorize' необходимо указать golden_tags_path")
        if not sku:
            raise ValueError("Для режима 'vectorize' необходимо указать sku")

        print(f"\nVECTORIZE для SKU={sku} ...")

        tags = run_vector_pipeline_for_sku(
            csv_path=str(csv_path),
            golden_tags_path=str(golden_tags_path),
            sku=sku,
            max_chars=max_chars,
            max_reviews=max_reviews,
            min_review_length=min_length,
            max_tags=6,
            min_reviews_per_sku=1,
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )

        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТЫ VECTORIZE (одна SKU):")
        print("=" * 80)
        print(f"SKU: {sku}")
        print(f"Теги ({len(tags)}): {', '.join(tags) if tags else 'нет тегов'}")
        print("=" * 80)

        return tags

    else:
        raise ValueError(f"Неизвестный режим mode={mode}, ожидается 'llm' или 'vectorize'")