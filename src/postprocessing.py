"""Модуль для постобработки тегов, полученных от LLM."""

import re
from typing import List, Set


def postprocess_tags(
    llm_response: str,
    golden_tags: List[str],
    max_tags: int = 6,
) -> List[str]:
    """
    Постобрабатывает теги, полученные от LLM.

    Выполняет следующие операции:
    - Проверяет, что все теги входят в GOLDEN TAGS
    - Убирает дубликаты
    - Ограничивает максимум max_tags тегами
    - Fallback: если LLM вернул пусто → возвращает пустой список

    Args:
        llm_response: Ответ от LLM в виде строки.
        golden_tags: Список допустимых GOLDEN TAGS.
        max_tags: Максимальное количество тегов для возврата. По умолчанию 6.

    Returns:
        Список валидных тегов (максимум max_tags).

    Examples:
        >>> golden_tags = ["качество", "размер", "цена"]
        >>> postprocess_tags("качество, размер, качество", golden_tags, max_tags=2)
        ['качество', 'размер']
        >>> postprocess_tags("", golden_tags)
        []
        >>> postprocess_tags("несуществующий_тег", golden_tags)
        []
    """
    # Fallback: если LLM вернул пусто → возвращаем пустой список
    if not llm_response or not llm_response.strip():
        return []

    # Нормализуем список GOLDEN TAGS (приводим к нижнему регистру для сравнения)
    golden_tags_lower = {tag.lower().strip(): tag for tag in golden_tags}

    # Извлекаем теги из ответа LLM
    # Разделяем по запятым, точкам с запятой, переносам строк
    raw_tags = re.split(r'[,;\n]', llm_response)

    # Очищаем и нормализуем теги
    processed_tags: List[str] = []
    seen_tags: Set[str] = set()

    for raw_tag in raw_tags:
        # Убираем пробелы и приводим к нижнему регистру
        tag_clean = raw_tag.strip().lower()

        # Пропускаем пустые теги
        if not tag_clean:
            continue

        # Проверяем, что тег входит в GOLDEN TAGS
        if tag_clean in golden_tags_lower:
            # Используем оригинальный тег из golden_tags (с правильным регистром)
            original_tag = golden_tags_lower[tag_clean]

            # Убираем дубликаты
            if original_tag not in seen_tags:
                processed_tags.append(original_tag)
                seen_tags.add(original_tag)

                # Ограничиваем максимум max_tags тегами
                if len(processed_tags) >= max_tags:
                    break

    return processed_tags

