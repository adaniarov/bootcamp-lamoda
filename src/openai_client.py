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
                "в переменной окружения OPENAI_API_KEY. "
                "Создайте файл .env в корне проекта с содержимым: OPENAI_API_KEY=ваш-ключ"
            )

        # Проверяем формат ключа (базовая проверка)
        api_key_clean = api_key.strip()
        if not (api_key_clean.startswith("sk-") or api_key_clean.startswith("sk-proj-")):
            raise ValueError(
                f"Неверный формат API ключа. Ключ должен начинаться с 'sk-' или 'sk-proj-'. "
                f"Получен ключ, начинающийся с: {api_key_clean[:10]}..."
            )

        self.client = OpenAI(api_key=api_key_clean)

    def generate(self, prompt: str) -> str:
        """
        Генерирует ответ на основе промпта.

        Args:
            prompt: Текст промпта

        Returns:
            Ответ от LLM

        Raises:
            Exception: При ошибке обращения к API (включая 403 Forbidden)
        """
        try:
            # Параметр store может быть не поддерживаемым в некоторых версиях API
            # Пробуем сначала с store, если ошибка - без него
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    store=self.store,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as e:
                # Если ошибка связана с store, пробуем без него
                if "store" in str(e).lower() or "403" in str(e):
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                else:
                    raise

            return completion.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            # Более информативные сообщения об ошибках
            if "403" in error_msg or "Forbidden" in error_msg:
                raise Exception(
                    f"Ошибка доступа к OpenAI API (403 Forbidden). "
                    f"Проверьте:\n"
                    f"1. Правильность API ключа в переменной окружения OPENAI_API_KEY\n"
                    f"2. Что ключ имеет доступ к модели {self.model}\n"
                    f"3. Что ключ не истек и активен\n"
                    f"4. Формат ключа: должен начинаться с 'sk-' или 'sk-proj-'"
                ) from e
            elif "401" in error_msg or "Unauthorized" in error_msg:
                raise Exception(
                    f"Ошибка аутентификации (401 Unauthorized). "
                    f"Проверьте правильность API ключа в переменной окружения OPENAI_API_KEY"
                ) from e
            else:
                raise Exception(f"Ошибка при вызове OpenAI API: {error_msg}") from e

