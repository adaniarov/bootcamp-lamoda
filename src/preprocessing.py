"""Модуль для предобработки отзывов."""

from typing import List, Optional


def prepare_reviews(
    reviews: List[str],
    max_chars: int = 500,
    max_reviews: int = 10,
    min_length: int = 10,
) -> List[str]:
    """
    Предобрабатывает отзывы:
    - Обрезает каждый отзыв до max_chars символов
    - Берет не более max_reviews отзывов
    - Удаляет пустые и слишком короткие отзывы

    Args:
        reviews: Список отзывов
        max_chars: Максимальная длина отзыва в символах
        max_reviews: Максимальное количество отзывов
        min_length: Минимальная длина отзыва (отзывы короче будут удалены)

    Returns:
        Список предобработанных отзывов
    """
    # Удаляем пустые и слишком короткие отзывы
    filtered_reviews = [
        review.strip()
        for review in reviews
        if review and isinstance(review, str) and len(review.strip()) >= min_length
    ]

    # Обрезаем каждый отзыв до max_chars символов
    truncated_reviews = [review[:max_chars] for review in filtered_reviews]

    # Берем не более max_reviews отзывов
    limited_reviews = truncated_reviews[:max_reviews]

    return limited_reviews

