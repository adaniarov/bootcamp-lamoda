"""Модуль для построения промптов для LLM."""

from typing import Dict, List, Optional


def get_golden_tags_for_product(product_name: str, golden_tags: Dict[str, str]) -> Optional[str]:
    """
    Получает golden_tag для товара по его названию.

    Args:
        product_name: Название товара
        golden_tags: Словарь {name: golden_tag}

    Returns:
        golden_tag или None, если не найден
    """
    return golden_tags.get(product_name)


def build_prompt(
    product_name: str,
    reviews: List[str],
    golden_tag: Optional[str] = None,
) -> str:
    """
    Строит промпт для LLM на основе отзывов и golden_tag.

    Args:
        product_name: Название товара
        reviews: Список отзывов
        golden_tag: Золотой тег для товара (опционально)

    Returns:
        Текст промпта
    """
    reviews_text = "\n".join([f"- {review}" for review in reviews])

    prompt = f"""Проанализируй отзывы о товаре "{product_name}" и определи основные теги/характеристики.

Отзывы:
{reviews_text}
"""

    if golden_tag:
        prompt += f"\nЗолотой тег (эталонный): {golden_tag}\n"

    prompt += """
На основе анализа отзывов, определи основные теги для этого товара. 
Ответ должен быть в формате: название_товара-тег1,тег2,тег3
Теги должны быть краткими и отражать основные характеристики товара, упомянутые в отзывах.
"""

    return prompt

