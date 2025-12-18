"""Базовый интерфейс для LLM клиентов."""

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Базовый класс для LLM клиентов."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Генерирует ответ на основе промпта.

        Args:
            prompt: Текст промпта

        Returns:
            Ответ от LLM
        """
        pass


class LLMClient(BaseLLMClient):
    """Обертка для совместимости с существующим кодом."""

    def __init__(self, client: BaseLLMClient):
        """
        Args:
            client: Реализация BaseLLMClient
        """
        self._client = client

    def generate(self, prompt: str) -> str:
        """Генерирует ответ на основе промпта."""
        return self._client.generate(prompt)

