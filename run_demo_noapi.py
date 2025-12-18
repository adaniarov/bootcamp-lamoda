"""Скрипт для запуска демонстрации БЕЗ использования реального LLM API.

Использует mock клиент для проверки всего процесса без обращения к OpenAI API.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.llm_client import BaseLLMClient
from src.pipeline import run_pipeline_for_file


class MockLLMClient(BaseLLMClient):
    """Mock клиент для тестирования без реального API."""

    def __init__(self):
        """Инициализирует mock клиент."""
        print("  [MOCK] Используется mock LLM клиент (без реального API)")

    def generate(self, prompt: str) -> str:
        """
        Генерирует mock ответ на основе промпта в формате JSON.

        Извлекает golden_tags из промпта и возвращает их в JSON формате,
        имитируя работу LLM.

        Args:
            prompt: Текст промпта

        Returns:
            Mock ответ в формате JSON: {"top_tags": ["тег1", "тег2", ...]}
        """
        import json
        
        # Извлекаем теги из промпта (ищем в теге <tags>)
        tags_match = re.search(r"<tags>(.+?)</tags>", prompt, re.DOTALL)
        
        if tags_match:
            tags_text = tags_match.group(1).strip()
            # Разбиваем теги по запятым
            all_tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]

            # Имитируем выбор тегов: берем первые 6 тегов из списка
            # (в реальности LLM выбирал бы на основе отзывов)
            selected_tags = all_tags[: min(6, len(all_tags))]

            # Возвращаем в формате JSON
            return json.dumps({"top_tags": selected_tags}, ensure_ascii=False, indent=2)

        # Если не удалось извлечь теги, возвращаем дефолтный ответ в JSON формате
        default_tags = ["качество", "размер", "материал", "цена", "комфорт", "стиль"]
        return json.dumps({"top_tags": default_tags}, ensure_ascii=False, indent=2)


def main(
    csv_path: str | Path = "data/interim/one_example.csv",
    golden_tags_path: str | Path | None = None,
    max_chars: int = 500,
    max_reviews: int = 10,
    min_length: int = 10,
    limit: Optional[int] = None,
):
    """
    Запускает демонстрацию работы БЕЗ реального LLM API.

    Args:
        csv_path: Путь к CSV файлу с отзывами
        golden_tags_path: Путь к JSON файлу с GOLDEN_TAGS (опционально)
        max_chars: Максимальная длина отзыва в символах
        max_reviews: Максимальное количество отзывов
        min_length: Минимальная длина отзыва
        limit: Количество товаров для обработки (для демонстрации)
    """
    # Преобразуем пути в Path объекты
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        project_root = Path(__file__).parent
        csv_path = project_root / csv_path

    if golden_tags_path:
        golden_tags_path = Path(golden_tags_path)
        if not golden_tags_path.is_absolute():
            project_root = Path(__file__).parent
            golden_tags_path = project_root / golden_tags_path

    print("=" * 80)
    print("ЗАПУСК ДЕМОНСТРАЦИИ БЕЗ LLM API (MOCK РЕЖИМ)")
    print("=" * 80)
    print(f"CSV файл: {csv_path}")
    print(f"GOLDEN_TAGS файл: {golden_tags_path or 'не указан'}")
    print(f"Параметры: max_chars={max_chars}, max_reviews={max_reviews}, min_length={min_length}")
    print(f"Обработка товаров: {limit or 'все'}")
    print("=" * 80)

    # Инициализируем MOCK LLM клиент
    print("\nИнициализация MOCK LLM клиента...")
    llm_client = MockLLMClient()
    print("✓ Mock клиент инициализирован")

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
    print(f"\nЗапуск pipeline для обработки товаров (MOCK режим)...")
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
    print("РЕЗУЛЬТАТЫ (MOCK РЕЖИМ):")
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
    print("\n✓ Демонстрация завершена успешно (без использования реального API)")
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    # Запуск демонстрации БЕЗ API
    example_file = Path("data/interim/one_example.csv")
    golden_tags_file = Path("data/processed/golden_tags.json")

    if not example_file.exists():
        print("⚠ Внимание: Файл data/interim/one_example.csv не найден!")
        print("Сначала запустите ноутбук notebooks/get_one_example.ipynb для создания файла с примером.")
        print("Или используйте полный файл: data/raw/lamoda_reviews.csv")
        exit(1)

    # Проверяем наличие файла с golden_tags (не критично, если его нет)
    if not golden_tags_file.exists():
        print(f"⚠ Внимание: Файл {golden_tags_file} не найден, будет использоваться без golden_tags")
        golden_tags_path = None
    else:
        golden_tags_path = str(golden_tags_file)
        print(f"✓ Используется файл с golden_tags: {golden_tags_file}")

    main(
        csv_path=str(example_file),
        golden_tags_path=golden_tags_path,
        max_chars=500,
        max_reviews=10,
        min_length=10,
        limit=None,
    )

