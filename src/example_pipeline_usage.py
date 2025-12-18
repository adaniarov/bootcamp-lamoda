"""Пример использования pipeline для обработки файла."""

from typing import Dict, List

from src.llm_client import LLMClient
from src.pipeline import run_pipeline_for_file, run_pipeline_for_sku


class MockLLMClient:
    """Пример реализации LLM клиента для тестирования."""

    def generate(self, prompt: str) -> str:
        """
        Генерирует ответ на основе промпта (мок-реализация).

        Args:
            prompt: Текст промпта для LLM.

        Returns:
            Ответ от LLM в виде строки.
        """
        # Простая мок-реализация для демонстрации
        if "качество" in prompt.lower():
            return "качество, размер"
        return "тег1, тег2"


def example_run_pipeline_for_file():
    """Пример запуска pipeline для всего файла."""
    # Создаем мок-клиент LLM
    llm_client: LLMClient = MockLLMClient()

    # Подготавливаем словари GOLDEN TAGS
    name_to_tags: Dict[str, List[str]] = {
        "Футболка": ["качество", "размер", "материал", "цена", "цвет"],
        "Джинсы": ["качество", "размер", "посадка", "цена"],
        "Носки": ["качество", "размер", "материал"],
    }

    subtype_to_tags: Dict[str, List[str]] = {
        "TEE-SHIRTS & POLOS": ["качество", "размер", "материал"],
        "JEANS": ["качество", "размер", "посадка"],
        "SOCKS & TIGHTS": ["качество", "размер"],
    }

    type_to_tags: Dict[str, List[str]] = {
        "Clothes": ["качество", "размер", "цена"],
        "Shoes": ["качество", "размер", "комфорт"],
    }

    # Запускаем pipeline для файла
    results_df = run_pipeline_for_file(
        csv_path="data/raw/lamoda_reviews.csv",
        llm_client=llm_client,
        name_to_tags=name_to_tags,
        subtype_to_tags=subtype_to_tags,
        type_to_tags=type_to_tags,
        output_path="data/processed/llm_tags_results.csv",
        max_chars=500,
        max_reviews=50,
        min_review_length=10,
        max_tags=6,
        limit_skus=10,  # Ограничиваем для тестирования
        skip_errors=True,
    )

    print(f"\nОбработано {len(results_df)} SKU")
    print(f"SKU с тегами: {len(results_df[results_df['num_tags'] > 0])}")
    print("\nПервые результаты:")
    print(results_df.head())


def example_run_pipeline_for_sku():
    """Пример запуска pipeline для одного SKU."""
    # Создаем мок-клиент LLM
    llm_client: LLMClient = MockLLMClient()

    # Подготавливаем словари GOLDEN TAGS
    name_to_tags: Dict[str, List[str]] = {
        "Футболка": ["качество", "размер", "материал"],
    }

    # Запускаем pipeline для одного SKU
    tags = run_pipeline_for_sku(
        csv_path="data/raw/lamoda_reviews.csv",
        sku="MP002XW0O0SI",  # Пример SKU
        llm_client=llm_client,
        name_to_tags=name_to_tags,
        max_chars=500,
        max_reviews=50,
        max_tags=6,
    )

    print(f"\nПолученные теги для SKU: {tags}")


if __name__ == "__main__":
    print("=== Пример 1: Обработка всего файла ===")
    example_run_pipeline_for_file()

    print("\n=== Пример 2: Обработка одного SKU ===")
    example_run_pipeline_for_sku()

