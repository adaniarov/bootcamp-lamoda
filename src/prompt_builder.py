"""Модуль для построения промптов для LLM."""

from typing import Dict, List, Optional


def build_prompt(
    reviews: List[str],
    golden_tags: List[str],
    product_name: Optional[str] = None,
    product_subtype: Optional[str] = None,
    product_type: Optional[str] = None,
    custom_prompt_template: Optional[str] = None,
) -> str:
    """
    Строит промпт для LLM на основе отзывов и GOLDEN TAGS.

    Args:
        reviews: Список обработанных отзывов по SKU.
        golden_tags: Список GOLDEN TAGS для данного продукта.
        product_name: Название продукта (опционально).
        product_subtype: Подтип продукта (опционально).
        product_type: Тип продукта (опционально).
        custom_prompt_template: Кастомный шаблон промпта. Если не указан,
            используется стандартный шаблон.

    Returns:
        Готовый промпт для LLM.

    Examples:
        >>> reviews = ["Отличное качество", "Хороший размер"]
        >>> tags = ["качество", "размер", "цена"]
        >>> prompt = build_prompt(reviews, tags, product_name="Футболка")
        >>> len(prompt) > 0
        True
    """
    if not reviews:
        raise ValueError("Список отзывов не может быть пустым")

    if not golden_tags:
        raise ValueError("Список GOLDEN TAGS не может быть пустым")

    # Формируем информацию о продукте
    product_info_parts = []
    if product_name:
        product_info_parts.append(f"Название продукта: {product_name}")
    if product_subtype:
        product_info_parts.append(f"Подтип: {product_subtype}")
    if product_type:
        product_info_parts.append(f"Тип: {product_type}")

    product_info = "\n".join(product_info_parts) if product_info_parts else "Информация о продукте не указана"

    # Формируем список отзывов
    reviews_text = "\n".join([f"{i+1}. {review}" for i, review in enumerate(reviews, 1)])

    # Формируем список GOLDEN TAGS
    tags_text = ", ".join(golden_tags)

    # Используем кастомный шаблон или стандартный
    if custom_prompt_template:
        prompt = custom_prompt_template.format(
            product_info=product_info,
            reviews=reviews_text,
            golden_tags=tags_text,
        )
    else:
        prompt = f"""Проанализируй отзывы о продукте и выбери только релевантные теги из списка GOLDEN TAGS.

{product_info}

Отзывы о продукте:
{reviews_text}

Доступные GOLDEN TAGS (выбери только релевантные):
{tags_text}

Инструкция:
1. Проанализируй каждый отзыв
2. Выбери только те теги из списка GOLDEN TAGS, которые действительно упоминаются или подразумеваются в отзывах
3. Верни только список выбранных тегов, разделенных запятыми
4. Если ни один тег не релевантен, верни пустую строку
5. Не добавляй теги, которых нет в списке GOLDEN TAGS

Выбранные теги:"""

    return prompt


def get_golden_tags_for_product(
    product_name: Optional[str],
    product_subtype: Optional[str],
    product_type: Optional[str],
    name_to_tags: Dict[str, List[str]],
    subtype_to_tags: Dict[str, List[str]],
    type_to_tags: Dict[str, List[str]],
) -> List[str]:
    """
    Получает GOLDEN TAGS для продукта на основе иерархии name -> subtype -> type.

    Args:
        product_name: Название продукта.
        product_subtype: Подтип продукта.
        product_type: Тип продукта.
        name_to_tags: Словарь связки name -> список тегов.
        subtype_to_tags: Словарь связки subtype -> список тегов.
        type_to_tags: Словарь связки type -> список тегов.

    Returns:
        Список GOLDEN TAGS для продукта.

    Examples:
        >>> name_to_tags = {"Футболка": ["качество", "размер"]}
        >>> subtype_to_tags = {"TEE-SHIRTS": ["материал"]}
        >>> type_to_tags = {"Clothes": ["цена"]}
        >>> get_golden_tags_for_product(
        ...     "Футболка", "TEE-SHIRTS", "Clothes",
        ...     name_to_tags, subtype_to_tags, type_to_tags
        ... )
        ['качество', 'размер']
    """
    # Приоритет: name -> subtype -> type
    if product_name and product_name in name_to_tags:
        return name_to_tags[product_name]

    if product_subtype and product_subtype in subtype_to_tags:
        return subtype_to_tags[product_subtype]

    if product_type and product_type in type_to_tags:
        return type_to_tags[product_type]

    # Если ничего не найдено, возвращаем пустой список
    return []

