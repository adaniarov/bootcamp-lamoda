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

    # Формируем список отзывов (каждый отзыв на новой строке)
    reviews_text = "\n".join(reviews) if isinstance(reviews, list) else str(reviews)

    # Формируем список тегов (через запятую)
    tags_text = ", ".join(golden_tags) if isinstance(golden_tags, list) else str(golden_tags)

    # Подготавливаем значения для подстановки (защита от None)
    product_type_str = product_type or ""
    product_subtype_str = product_subtype or ""
    product_name_str = product_name or ""

    # Используем кастомный шаблон или стандартный
    if custom_prompt_template:
        prompt = custom_prompt_template.format(
            product_type=product_type_str,
            product_subtype=product_subtype_str,
            product_name=product_name_str,
            reviews=reviews_text,
            tags=tags_text,
        )
    else:
        prompt = f'''Ты — эксперт по анализу товаров для интернет-магазина Lamoda. Твоя задача — проанализировать информацию о товаре и выбрать ТОЧНО 6 самых релевантных тегов из предоставленного списка тегов.

ИНСТРУКЦИИ:
1. Ты ПОЛУЧАЕШЬ информацию о товаре и должен выбрать ровно 6 тегов из заданного списка.
2. КРИТЕРИИ ВЫБОРА:
   - Анализируй название товара, категорию и подкатегорию.
   - Внимательно изучи отзывы покупателей — они содержат ключевую информацию о реальных характеристиках товара.
   - Выбирай теги, которые наиболее часто упоминаются или подразумеваются в отзывах.
   - Учитывай практическое использование товара (сезонность, комфорт, особенности).
   - Приоритет отдается тегам, которые отражают ключевые преимущества товара по мнению покупателей.

3. СТРОГИЕ ПРАВИЛА:
   - Выбирай ТОЛЬКО из предоставленного списка тегов — никаких своих тегов!
   - ДОЛЖНО БЫТЬ ТОЧНО 6 тегов, не больше и не меньше.
   - Если в списке тегов меньше 6 вариантов — выбирай все доступные.
   - Не добавляй никаких пояснений, комментариев или дополнительного текста.
   - Вывод должен быть 100% валидным JSON.

4. ФОРМАТ ВЫВОДА (СТРОГО СЛЕДУЙ):
{{
  "top_tags": ["тег1", "тег2", "тег3", "тег4", "тег5", "тег6"]
}}

ВАЖНО для gpt-4o-mini:
- Следуй инструкциям точно — модель имеет ограниченные возможности.
- Не отклоняйся от формата вывода.
- Убедись, что все выбранные теги присутствуют в списке "tags".
- Проверь синтаксическую корректность JSON (кавычки, запятые, скобки).

ВХОДНЫЕ ДАННЫЕ:
<product_type>{product_type_str}</product_type>
<product_subtype>{product_subtype_str}</product_subtype>
<product_name>{product_name_str}</product_name>
<reviews>{reviews_text}</reviews>
<tags>{tags_text}</tags>
'''

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

