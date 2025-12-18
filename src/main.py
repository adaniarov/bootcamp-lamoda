"""Главный модуль для запуска демонстрации."""

from pathlib import Path

from .inference import run_inference
from .openai_client import OpenAILLMClient


def main(
    csv_path: str | Path = "data/raw/lamoda_reviews.csv",
    golden_tags_path: str | Path | None = None,
    max_chars: int = 500,
    max_reviews: int = 10,
    min_length: int = 10,
    limit: int = 5,
    model: str = "gpt-4o-mini",
):
    """
    Запускает демонстрацию работы с LLM для анализа отзывов.

    Args:
        csv_path: Путь к CSV файлу с отзывами
        golden_tags_path: Путь к JSON файлу с GOLDEN_TAGS (опционально)
        max_chars: Максимальная длина отзыва в символах
        max_reviews: Максимальное количество отзывов
        min_length: Минимальная длина отзыва
        limit: Количество товаров для обработки (для демонстрации)
        model: Название модели OpenAI
    """
    # Преобразуем пути в Path объекты
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        # Если путь относительный, делаем его относительно корня проекта
        project_root = Path(__file__).parent.parent
        csv_path = project_root / csv_path

    if golden_tags_path:
        golden_tags_path = Path(golden_tags_path)
        if not golden_tags_path.is_absolute():
            project_root = Path(__file__).parent.parent
            golden_tags_path = project_root / golden_tags_path

    print("=" * 80)
    print("ЗАПУСК ДЕМОНСТРАЦИИ LLM ИНФЕРЕНСА ДЛЯ ОТЗЫВОВ LAMODA")
    print("=" * 80)
    print(f"CSV файл: {csv_path}")
    print(f"GOLDEN_TAGS файл: {golden_tags_path or 'не указан'}")
    print(f"Параметры: max_chars={max_chars}, max_reviews={max_reviews}, min_length={min_length}")
    print(f"Обработка товаров: {limit}")
    print("=" * 80)

    # Инициализируем LLM клиент
    print("\nИнициализация OpenAI клиента...")
    llm_client = OpenAILLMClient(model=model)
    print("✓ Клиент инициализирован")

    # Запускаем инференс
    print(f"\nЗапуск инференса для {limit} товаров...")
    results = run_inference(
        csv_path=csv_path,
        llm_client=llm_client,
        golden_tags_path=golden_tags_path,
        max_chars=max_chars,
        max_reviews=max_reviews,
        min_length=min_length,
        limit=limit,
    )

    # Выводим результаты
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")

    print("\n" + "=" * 80)
    print(f"Обработано товаров: {len(results)}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    # Запуск демонстрации
    main()

