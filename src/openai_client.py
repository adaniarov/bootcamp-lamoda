"""OpenAI клиент для работы с LLM."""

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from .llm_client import BaseLLMClient

# Загружаем переменные окружения из .env файла
load_dotenv()


class OpenAILLMClient(BaseLLMClient):
    """Клиент для работы с OpenAI API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        store: bool = True,
    ):
        """
        Инициализирует OpenAI клиент.

        Args:
            api_key: API ключ OpenAI. Если не указан, берется из переменной окружения OPENAI_API_KEY
            model: Название модели
            store: Сохранять ли запросы в истории OpenAI
        """
        self.model = model
        self.store = store

        # Получаем API ключ из параметра или переменной окружения
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API ключ не найден. Укажите его в параметре api_key или "
                "в переменной окружения OPENAI_API_KEY"
            )

        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """
        Генерирует ответ на основе промпта.

        Args:
            prompt: Текст промпта

        Returns:
            Ответ от LLM
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            store=self.store,
            messages=[{"role": "user", "content": prompt}],
        )

        return completion.choices[0].message.content

