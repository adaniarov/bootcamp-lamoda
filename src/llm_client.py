"""Абстракция для работы с LLM клиентом."""

from abc import ABC, abstractmethod
from typing import Protocol


class LLMClient(Protocol):
    """Протокол для LLM клиента."""

    def generate(self, prompt: str) -> str:
        """
        Генерирует ответ на основе промпта.

        Args:
            prompt: Текст промпта для LLM.

        Returns:
            Ответ от LLM в виде строки.

        Raises:
            Exception: При ошибке обращения к LLM.
        """
        ...


class BaseLLMClient(ABC):
    """Базовый абстрактный класс для LLM клиента."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Генерирует ответ на основе промпта.

        Args:
            prompt: Текст промпта для LLM.

        Returns:
            Ответ от LLM в виде строки.

        Raises:
            Exception: При ошибке обращения к LLM.
        """
        pass

