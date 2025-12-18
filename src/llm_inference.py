"""Модуль для вызова LLM для одного товара."""

from typing import Dict, List, Optional

from .llm_client import BaseLLMClient
from .preprocessing import prepare_reviews
from .prompt_builder import build_prompt, get_golden_tags_for_product


def run_llm(
    sku: str,
    product_name: str,
    reviews: List[str],
    llm_client: BaseLLMClient,
    golden_tags: Optional[Dict[str, str]] = None,
    max_chars: int = 500,
    max_reviews: int = 10,
    min_length: int = 10,
) -> str:
    """
    Вызывает LLM для одного товара.

    Args:
        sku: SKU товара
        product_name: Название товара
        reviews: Список отзывов о товаре
        llm_client: Клиент для работы с LLM
        golden_tags: Словарь {name: golden_tag} (опционально)
        max_chars: Максимальная длина отзыва в символах
        max_reviews: Максимальное количество отзывов
        min_length: Минимальная длина отзыва

    Returns:
        Результат в формате: sku-name-tags
    """
    # Предобрабатываем отзывы
    processed_reviews = prepare_reviews(
        reviews=reviews,
        max_chars=max_chars,
        max_reviews=max_reviews,
        min_length=min_length,
    )

    if not processed_reviews:
        return f"{sku}-{product_name}-no_reviews"

    # Получаем golden_tag для товара
    golden_tag = None
    if golden_tags:
        golden_tag = get_golden_tags_for_product(product_name, golden_tags)

    # Строим промпт
    prompt = build_prompt(
        product_name=product_name,
        reviews=processed_reviews,
        golden_tag=golden_tag,
    )

    # Вызываем LLM
    response = llm_client.generate(prompt)

    # Формируем результат в формате sku-name-tags
    # Если ответ уже в нужном формате, используем его, иначе формируем сами
    if response.strip().startswith(f"{sku}-"):
        return response.strip()
    else:
        # Извлекаем теги из ответа или используем весь ответ как теги
        tags = response.strip()
        return f"{sku}-{product_name}-{tags}"

