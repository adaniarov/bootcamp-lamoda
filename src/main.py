"""Главный модуль для запуска демонстрации."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from .openai_client import OpenAILLMClient
from .pipeline import run_pipeline_for_file


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

    # Загружаем golden_tags из JSON файла, если указан
    name_to_tags: Optional[Dict[str, List[str]]] = None
    subtype_to_tags: Optional[Dict[str, List[str]]] = None
    type_to_tags: Optional[Dict[str, List[str]]] = None

    if golden_tags_path:
        print(f"\nЗагрузка GOLDEN_TAGS из {golden_tags_path}...")
        try:
            with open(golden_tags_path, "r", encoding="utf-8") as f:
                golden_tags_data = json.load(f)

            # Преобразуем JSON в нужный формат
            name_to_tags = {}
            for item in golden_tags_data:
                if "name" in item and "tags" in item:
                    name = item["name"]
                    tags = item["tags"]
                    # Если tags - список, используем как есть, иначе преобразуем
                    if isinstance(tags, list):
                        name_to_tags[name] = tags
                    elif isinstance(tags, str):
                        # Если строка, разбиваем по запятым
                        name_to_tags[name] = [t.strip() for t in tags.split(",") if t.strip()]

            print(f"✓ Загружено {len(name_to_tags)} записей golden_tags")
        except Exception as e:
            print(f"⚠ Ошибка при загрузке golden_tags: {e}")
            print("Продолжаем без golden_tags")

    # Запускаем pipeline
    print(f"\nЗапуск pipeline для обработки товаров...")
    results_df = run_pipeline_for_file(
        csv_path=str(csv_path),
        llm_client=llm_client,
        name_to_tags=name_to_tags,
        subtype_to_tags=subtype_to_tags,
        type_to_tags=type_to_tags,
        max_chars=max_chars,
        max_reviews=max_reviews,
        min_review_length=min_length,
        limit_skus=limit,
    )

    # Выводим результаты
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 80)
    for i, row in results_df.iterrows():
        print(f"{i+1}. SKU: {row['sku']}")
        print(f"   Название: {row['name']}")
        print(f"   Теги ({row['num_tags']}): {row['tags'] if row['tags'] else 'нет тегов'}")
        if row.get('error'):
            print(f"   ⚠ Ошибка: {row['error']}")
        print()

    print("=" * 80)
    print(f"Обработано товаров: {len(results_df)}")
    print(f"С тегами: {len(results_df[results_df['num_tags'] > 0])}")
    print(f"Без тегов: {len(results_df[results_df['num_tags'] == 0])}")
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    # Запуск демонстрации
    main()

