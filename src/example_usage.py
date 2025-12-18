"""Пример использования модулей для работы с LLM инференсом."""

from typing import List

from src.llm_client import LLMClient


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


def example_usage():
    """Пример использования функции run_inference."""
    from src.inference import run_inference

    # Создаем мок-клиент LLM
    llm_client: LLMClient = MockLLMClient()

    # Пример данных
    reviews: List[str] = [
        "Отличное качество товара, очень доволен покупкой!",
        "Хороший размер, подошел идеально.",
        "Качество на высоте, рекомендую.",
        "Размер немного мал, но качество хорошее.",
    ]

    # Словари связок для GOLDEN TAGS
    name_to_tags = {
        "Футболка": ["качество", "размер", "материал", "цена", "цвет"],
    }

    subtype_to_tags = {
        "TEE-SHIRTS & POLOS": ["качество", "размер", "материал"],
    }

    type_to_tags = {
        "Clothes": ["качество", "размер", "цена"],
    }

    # Выполняем инференс
    tags = run_inference(
        reviews=reviews,
        llm_client=llm_client,
        name_to_tags=name_to_tags,
        subtype_to_tags=subtype_to_tags,
        type_to_tags=type_to_tags,
        product_name="Футболка",
        product_subtype="TEE-SHIRTS & POLOS",
        product_type="Clothes",
        max_chars=500,
        max_reviews=50,
        min_review_length=10,
        max_tags=6,
    )

    print(f"Результат: {tags}")
    return tags


if __name__ == "__main__":
    example_usage()

