"""Модуль для постобработки тегов, полученных от LLM."""

import json
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

    # Пробуем извлечь теги из JSON формата
    raw_tags: List[str] = []
    
    # Пытаемся найти и распарсить JSON объект с top_tags
    try:
        # Ищем JSON объект в ответе
        json_match = re.search(r'\{[^{}]*"top_tags"[^{}]*\}', llm_response, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_str = json_match.group(0)
            parsed_json = json.loads(json_str)
            if "top_tags" in parsed_json and isinstance(parsed_json["top_tags"], list):
                raw_tags = [str(tag).strip() for tag in parsed_json["top_tags"] if tag]
    except (json.JSONDecodeError, KeyError, AttributeError):
        # Если не удалось распарсить JSON, пробуем извлечь теги из массива вручную
        try:
            # Ищем массив тегов в формате ["тег1", "тег2", ...]
            array_match = re.search(r'\[([^\]]+)\]', llm_response)
            if array_match:
                tags_content = array_match.group(1)
                raw_tags = re.findall(r'"([^"]+)"', tags_content)
        except Exception:
            pass
    
    # Если JSON не найден, пробуем старый формат (разделение по запятым)
    if not raw_tags:
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

