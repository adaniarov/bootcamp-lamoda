"""Главный модуль для выполнения инференса с использованием LLM."""

from typing import Dict, List, Optional

from .llm_client import LLMClient
from .llm_inference import run_llm
from .postprocessing import postprocess_tags
from .preprocessing import prepare_reviews
from .prompt_builder import build_prompt, get_golden_tags_for_product


def run_inference(
    reviews: List[str],
    llm_client: LLMClient,
    name_to_tags: Dict[str, List[str]],
    subtype_to_tags: Dict[str, List[str]],
    type_to_tags: Dict[str, List[str]],
    product_name: Optional[str] = None,
    product_subtype: Optional[str] = None,
    product_type: Optional[str] = None,
    max_chars: int = 500,
    max_reviews: int = 50,
    min_review_length: int = 10,
    max_tags: int = 6,
    custom_prompt_template: Optional[str] = None,
) -> List[str]:
    """
    Выполняет полный цикл инференса: препроцессинг -> промпт -> LLM -> постобработка.

    Args:
        reviews: Список исходных отзывов по SKU.
        llm_client: Клиент для работы с LLM.
        name_to_tags: Словарь связки name -> список тегов.
        subtype_to_tags: Словарь связки subtype -> список тегов.
        type_to_tags: Словарь связки type -> список тегов.
        product_name: Название продукта (опционально).
        product_subtype: Подтип продукта (опционально).
        product_type: Тип продукта (опционально).
        max_chars: Максимальная длина отзыва в символах. По умолчанию 500.
        max_reviews: Максимальное количество отзывов для обработки. По умолчанию 50.
        min_review_length: Минимальная длина отзыва для включения. По умолчанию 10.
        max_tags: Максимальное количество тегов для возврата. По умолчанию 6.
        custom_prompt_template: Кастомный шаблон промпта (опционально).

    Returns:
        Список валидных тегов (максимум max_tags).

    Raises:
        ValueError: Если reviews пустой или не удалось получить GOLDEN TAGS.
        Exception: При ошибке вызова LLM.

    Examples:
        >>> class MockLLMClient:
        ...     def generate(self, prompt: str) -> str:
        ...         return "качество, размер"
        >>> client = MockLLMClient()
        >>> reviews = ["Отличное качество", "Хороший размер"]
        >>> name_to_tags = {"Футболка": ["качество", "размер", "цена"]}
        >>> result = run_inference(
        ...     reviews, client, name_to_tags, {}, {},
        ...     product_name="Футболка"
        ... )
        >>> len(result) > 0
        True
    """
    # 1. Препроцессинг отзывов
    processed_reviews = prepare_reviews(
        reviews=reviews,
        max_chars=max_chars,
        max_reviews=max_reviews,
        min_review_length=min_review_length,
    )

    if not processed_reviews:
        return []

    # 2. Получаем GOLDEN TAGS для продукта
    golden_tags = get_golden_tags_for_product(
        product_name=product_name,
        product_subtype=product_subtype,
        product_type=product_type,
        name_to_tags=name_to_tags,
        subtype_to_tags=subtype_to_tags,
        type_to_tags=type_to_tags,
    )

    if not golden_tags:
        return []

    # 3. Строим промпт
    prompt = build_prompt(
        reviews=processed_reviews,
        golden_tags=golden_tags,
        product_name=product_name,
        product_subtype=product_subtype,
        product_type=product_type,
        custom_prompt_template=custom_prompt_template,
    )

    # 4. Вызываем LLM
    llm_response = run_llm(prompt=prompt, llm_client=llm_client)

    # 5. Постобработка тегов
    tags = postprocess_tags(
        llm_response=llm_response,
        golden_tags=golden_tags,
        max_tags=max_tags,
    )

    return tags

