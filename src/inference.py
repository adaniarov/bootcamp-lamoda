"""Модуль для запуска инференса."""

from pathlib import Path
from typing import List, Optional

from .llm_client import BaseLLMClient
from .pipeline import run_pipeline_for_file


def run_inference(
    csv_path: str | Path,
    llm_client: BaseLLMClient,
    golden_tags_path: Optional[str | Path] = None,
    max_chars: int = 500,
    max_reviews: int = 10,
    min_length: int = 10,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Запускает инференс для всех товаров в датасете.

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
    return run_pipeline_for_file(
        csv_path=csv_path,
        llm_client=llm_client,
        golden_tags_path=golden_tags_path,
        max_chars=max_chars,
        max_reviews=max_reviews,
        min_length=min_length,
        limit=limit,
    )

