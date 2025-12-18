"""Модуль для препроцессинга отзывов."""

from typing import List


def prepare_reviews(
    reviews: List[str],
    max_chars: int = 500,
    max_reviews: int = 50,
    min_review_length: int = 10,
) -> List[str]:
    """
    Подготавливает отзывы для обработки.

    Выполняет следующие операции:
    - Обрезает каждый отзыв до max_chars символов
    - Берет не более max_reviews отзывов
    - Удаляет пустые и слишком короткие отзывы

    Args:
        reviews: Список исходных отзывов.
        max_chars: Максимальная длина отзыва в символах. По умолчанию 500.
        max_reviews: Максимальное количество отзывов для обработки. По умолчанию 50.
        min_review_length: Минимальная длина отзыва для включения. По умолчанию 10.

    Returns:
        Список обработанных отзывов.

    Examples:
        >>> reviews = ["Очень хороший товар!", "Плохо", "Отличное качество"]
        >>> prepare_reviews(reviews, max_chars=20, max_reviews=2, min_review_length=5)
        ['Очень хороший товар!', 'Отличное качество']
    """
    if not reviews:
        return []

    # Удаляем пустые и слишком короткие отзывы
    filtered_reviews = [
        review.strip()
        for review in reviews
        if review and isinstance(review, str) and len(review.strip()) >= min_review_length
    ]

    if not filtered_reviews:
        return []

    # Обрезаем каждый отзыв до max_chars символов
    truncated_reviews = [review[:max_chars] for review in filtered_reviews]

    # Берем не более max_reviews отзывов
    result = truncated_reviews[:max_reviews]

    return result

