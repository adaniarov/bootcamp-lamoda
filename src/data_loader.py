"""Модуль для загрузки данных из CSV файла и GOLDEN_TAGS."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Загружает датасет отзывов из CSV файла.

    Args:
        file_path: Путь к CSV файлу с отзывами

    Returns:
        DataFrame с колонками: comment_id, product_sku, comment_text, name, good_type, good_subtype
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    df = pd.read_csv(file_path)
    return df


def load_golden_tags(file_path: str | Path) -> Dict[str, str]:
    """
    Загружает GOLDEN_TAGS из JSON файла.

    Ожидаемый формат JSON:
    [
        {"name": "название товара", "tags": "тег"},
        ...
    ]

    Args:
        file_path: Путь к JSON файлу с GOLDEN_TAGS

    Returns:
        Словарь {name: tags}
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Преобразуем список словарей в словарь name -> golden_tag
    golden_tags = {}
    for item in data:
        if "name" in item and "tags" in item:
            golden_tags[item["name"]] = item["tags"]

    return golden_tags


def load_golden_tags_from_dict(data: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Загружает GOLDEN_TAGS из словаря/списка словарей.

    Args:
        data: Список словарей с ключами "name" и "tags"

    Returns:
        Словарь {name: tags}
    """
    golden_tags = {}
    for item in data:
        if "name" in item and "golden_tag" in item:
            golden_tags[item["name"]] = item["golden_tag"]
    return golden_tags

