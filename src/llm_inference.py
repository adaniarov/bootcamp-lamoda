"""Модуль для вызова LLM."""

from typing import Optional

from src.llm_client import LLMClient


def run_llm(
    prompt: str,
    llm_client: LLMClient,
    max_retries: int = 3,
) -> str:
    """
    Вызывает LLM с заданным промптом.

    Args:
        prompt: Промпт для LLM.
        llm_client: Клиент для работы с LLM (должен иметь метод generate).
        max_retries: Максимальное количество попыток при ошибке. По умолчанию 3.

    Returns:
        Ответ от LLM в виде строки.

    Raises:
        ValueError: Если prompt пустой.
        Exception: Если все попытки вызова LLM завершились ошибкой.

    Examples:
        >>> class MockLLMClient:
        ...     def generate(self, prompt: str) -> str:
        ...         return "тег1, тег2"
        >>> client = MockLLMClient()
        >>> result = run_llm("Выбери теги", client)
        >>> "тег" in result.lower()
        True
    """
    if not prompt or not prompt.strip():
        raise ValueError("Промпт не может быть пустым")

    last_error = None
    for attempt in range(max_retries):
        try:
            response = llm_client.generate(prompt)
            return response.strip()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                continue
            else:
                raise Exception(
                    f"Не удалось получить ответ от LLM после {max_retries} попыток"
                ) from last_error

    # Этот код не должен выполняться, но для типизации
    raise Exception("Неожиданная ошибка при вызове LLM") from last_error

