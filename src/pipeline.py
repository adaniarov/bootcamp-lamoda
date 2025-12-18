"""Модуль для прогона LLM по всему датасету."""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .data_loader import load_dataset, load_golden_tags
from .llm_client import BaseLLMClient
from .llm_inference import run_llm


def run_pipeline_for_sku(
    df: pd.DataFrame,
    sku: str,
    llm_client: BaseLLMClient,
    golden_tags: Optional[Dict[str, str]] = None,
    max_chars: int = 500,
    max_reviews: int = 10,
    min_length: int = 10,
) -> str:
    """
    Запускает LLM для одного товара по SKU.

    Args:
        df: DataFrame с отзывами
        sku: SKU товара
        llm_client: Клиент для работы с LLM
        golden_tags: Словарь {name: golden_tag} (опционально)
        max_chars: Максимальная длина отзыва в символах
        max_reviews: Максимальное количество отзывов
        min_length: Минимальная длина отзыва

    Returns:
        Результат в формате: sku-name-tags
    """
    # Фильтруем данные по SKU
    sku_data = df[df["product_sku"] == sku]

    if sku_data.empty:
        return f"{sku}-not_found-no_data"

    # Получаем название товара
    product_name = sku_data["name"].iloc[0] if "name" in sku_data.columns else "Unknown"

    # Получаем отзывы
    reviews = sku_data["comment_text"].dropna().tolist()

    # Вызываем LLM
    result = run_llm(
        sku=sku,
        product_name=product_name,
        reviews=reviews,
        llm_client=llm_client,
        golden_tags=golden_tags,
        max_chars=max_chars,
        max_reviews=max_reviews,
        min_length=min_length,
    )

    return result


def run_pipeline_for_file(
    csv_path: str | Path,
    llm_client: BaseLLMClient,
    golden_tags_path: Optional[str | Path] = None,
    max_chars: int = 500,
    max_reviews: int = 10,
    min_length: int = 10,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Запускает LLM для всех товаров в датасете.

    Args:
        csv_path: Путь к CSV файлу с отзывами
        llm_client: Клиент для работы с LLM
        golden_tags_path: Путь к JSON файлу с GOLDEN_TAGS (опционально)
        max_chars: Максимальная длина отзыва в символах
        max_reviews: Максимальное количество отзывов
        min_length: Минимальная длина отзыва
        limit: Ограничение количества товаров для обработки (опционально)

    Returns:
        Список результатов в формате: sku-name-tags
    """
    # Загружаем данные
    df = load_dataset(csv_path)

    # Загружаем golden_tags, если указан путь
    golden_tags = None
    if golden_tags_path:
        golden_tags = load_golden_tags(golden_tags_path)

    # Получаем уникальные SKU
    unique_skus = df["product_sku"].unique()

    # Ограничиваем количество, если указано
    if limit:
        unique_skus = unique_skus[:limit]

    # Обрабатываем каждый товар
    results = []
    for sku in unique_skus:
        result = run_pipeline_for_sku(
            df=df,
            sku=sku,
            llm_client=llm_client,
            golden_tags=golden_tags,
            max_chars=max_chars,
            max_reviews=max_reviews,
            min_length=min_length,
        )
        results.append(result)

    return results

