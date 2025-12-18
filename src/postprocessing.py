"""Модуль для постобработки результатов LLM."""

from typing import List


def postprocess_tags(result: str) -> dict:
    """
    Парсит результат в формате sku-name-tags в словарь.

    Args:
        result: Результат в формате "sku-name-tags"

    Returns:
        Словарь с ключами: sku, name, tags
    """
    parts = result.split("-", 2)
    if len(parts) == 3:
        return {"sku": parts[0], "name": parts[1], "tags": parts[2]}
    elif len(parts) == 2:
        return {"sku": parts[0], "name": parts[1], "tags": ""}
    else:
        return {"sku": parts[0], "name": "", "tags": ""}


def postprocess_tags_batch(results: List[str]) -> List[dict]:
    """
    Парсит список результатов в список словарей.

    Args:
        results: Список результатов в формате "sku-name-tags"

    Returns:
        Список словарей с ключами: sku, name, tags
    """
    return [postprocess_tags(result) for result in results]

