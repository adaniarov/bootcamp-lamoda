"""Модуль для запуска полного pipeline обработки отзывов."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_loader import load_dataset, load_golden_tags_from_dict
from .inference import run_inference
from .llm_client import LLMClient

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline_for_file(
    csv_path: str,
    llm_client: LLMClient,
    name_to_tags: Optional[Dict[str, List[str]]] = None,
    subtype_to_tags: Optional[Dict[str, List[str]]] = None,
    type_to_tags: Optional[Dict[str, List[str]]] = None,
    output_path: Optional[str] = None,
    max_chars: int = 500,
    max_reviews: int = 50,
    min_review_length: int = 10,
    max_tags: int = 6,
    min_reviews_per_sku: int = 1,
    custom_prompt_template: Optional[str] = None,
    skip_errors: bool = True,
    limit_skus: Optional[int] = None,
) -> pd.DataFrame:
    """
    Запускает полный pipeline обработки отзывов для файла.

    Выполняет следующие шаги:
    1. Загружает данные из CSV
    2. Группирует отзывы по SKU
    3. Для каждого SKU запускает инференс через LLM
    4. Сохраняет результаты в CSV (если указан output_path)

    Args:
        csv_path: Путь к CSV файлу с отзывами.
        llm_client: Клиент для работы с LLM.
        name_to_tags: Словарь связки name -> список тегов.
        subtype_to_tags: Словарь связки subtype -> список тегов.
        type_to_tags: Словарь связки type -> список тегов.
        output_path: Путь для сохранения результатов. Если None, результаты не сохраняются.
        max_chars: Максимальная длина отзыва в символах. По умолчанию 500.
        max_reviews: Максимальное количество отзывов для обработки. По умолчанию 50.
        min_review_length: Минимальная длина отзыва для включения. По умолчанию 10.
        max_tags: Максимальное количество тегов для возврата. По умолчанию 6.
        min_reviews_per_sku: Минимальное количество отзывов для включения SKU. По умолчанию 1.
        custom_prompt_template: Кастомный шаблон промпта (опционально).
        skip_errors: Пропускать ошибки для отдельных SKU и продолжать обработку. По умолчанию True.
        limit_skus: Ограничить количество обрабатываемых SKU (для тестирования). По умолчанию None.

    Returns:
        DataFrame с результатами обработки. Колонки: sku, name, subtype, type, 
        tags (строка с тегами через запятую), num_tags, num_reviews, error (если была ошибка).

    Raises:
        FileNotFoundError: Если входной файл не найден.
        ValueError: Если данные не могут быть загружены.

    Examples:
        >>> class MockLLMClient:
        ...     def generate(self, prompt: str) -> str:
        ...         return "качество, размер"
        >>> client = MockLLMClient()
        >>> name_tags = {"Футболка": ["качество", "размер"]}
        >>> results = run_pipeline_for_file(
        ...     "data/raw/lamoda_reviews.csv",
        ...     client,
        ...     name_to_tags=name_tags,
        ...     limit_skus=5
        ... )
        >>> len(results) > 0
        True
    """
    logger.info(f"Начало обработки файла: {csv_path}")

    # Загружаем данные
    try:
        sku_data = load_dataset(
            csv_path=csv_path,
            min_reviews_per_sku=min_reviews_per_sku,
        )
        logger.info(f"Загружено {len(sku_data)} SKU с отзывами")
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        raise

    # Загружаем GOLDEN TAGS
    name_tags, subtype_tags, type_tags = load_golden_tags_from_dict(
        name_to_tags=name_to_tags,
        subtype_to_tags=subtype_to_tags,
        type_to_tags=type_to_tags,
    )
    logger.info(
        f"Загружено GOLDEN TAGS: name={len(name_tags)}, "
        f"subtype={len(subtype_tags)}, type={len(type_tags)}"
    )

    # Ограничиваем количество SKU для тестирования
    if limit_skus is not None:
        sku_list = list(sku_data.keys())[:limit_skus]
        sku_data = {sku: sku_data[sku] for sku in sku_list}
        logger.info(f"Ограничено до {limit_skus} SKU для обработки")

    # Обрабатываем каждый SKU
    results: List[Dict[str, Any]] = []
    total_skus = len(sku_data)
    processed = 0
    errors = 0

    for sku, product_data in sku_data.items():
        processed += 1
        logger.info(
            f"Обработка SKU {processed}/{total_skus}: {sku} "
            f"({product_data['num_reviews']} отзывов)"
        )

        try:
            # Запускаем инференс
            tags = run_inference(
                reviews=product_data["reviews"],
                llm_client=llm_client,
                name_to_tags=name_tags,
                subtype_to_tags=subtype_tags,
                type_to_tags=type_tags,
                product_name=product_data["name"],
                product_subtype=product_data["subtype"],
                product_type=product_data["type"],
                max_chars=max_chars,
                max_reviews=max_reviews,
                min_review_length=min_review_length,
                max_tags=max_tags,
                custom_prompt_template=custom_prompt_template,
            )

            results.append(
                {
                    "sku": sku,
                    "name": product_data["name"],
                    "subtype": product_data["subtype"],
                    "type": product_data["type"],
                    "tags": ", ".join(tags),
                    "num_tags": len(tags),
                    "num_reviews": product_data["num_reviews"],
                    "error": None,
                }
            )

            logger.info(f"✓ {sku}: получено {len(tags)} тегов")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"✗ Ошибка для {sku}: {error_msg}")

            if skip_errors:
                results.append(
                    {
                        "sku": sku,
                        "name": product_data["name"],
                        "subtype": product_data["subtype"],
                        "type": product_data["type"],
                        "tags": "",
                        "num_tags": 0,
                        "num_reviews": product_data["num_reviews"],
                        "error": error_msg,
                    }
                )
                errors += 1
            else:
                raise

    # Создаем DataFrame с результатами
    df_results = pd.DataFrame(results)

    # Сохраняем результаты, если указан путь
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Результаты сохранены в: {output_path}")

    # Выводим статистику
    logger.info(
        f"Обработка завершена: {processed} SKU обработано, "
        f"{errors} ошибок, {len(df_results[df_results['num_tags'] > 0])} SKU с тегами"
    )

    return df_results


def run_pipeline_for_sku(
    csv_path: str,
    sku: str,
    llm_client: LLMClient,
    name_to_tags: Optional[Dict[str, List[str]]] = None,
    subtype_to_tags: Optional[Dict[str, List[str]]] = None,
    type_to_tags: Optional[Dict[str, List[str]]] = None,
    max_chars: int = 500,
    max_reviews: int = 50,
    min_review_length: int = 10,
    max_tags: int = 6,
    custom_prompt_template: Optional[str] = None,
) -> List[str]:
    """
    Запускает pipeline для одного конкретного SKU.

    Args:
        csv_path: Путь к CSV файлу с отзывами.
        sku: SKU продукта для обработки.
        llm_client: Клиент для работы с LLM.
        name_to_tags: Словарь связки name -> список тегов.
        subtype_to_tags: Словарь связки subtype -> список тегов.
        type_to_tags: Словарь связки type -> список тегов.
        max_chars: Максимальная длина отзыва в символах. По умолчанию 500.
        max_reviews: Максимальное количество отзывов для обработки. По умолчанию 50.
        min_review_length: Минимальная длина отзыва для включения. По умолчанию 10.
        max_tags: Максимальное количество тегов для возврата. По умолчанию 6.
        custom_prompt_template: Кастомный шаблон промпта (опционально).

    Returns:
        Список тегов для данного SKU.

    Raises:
        ValueError: Если SKU не найден в файле.

    Examples:
        >>> class MockLLMClient:
        ...     def generate(self, prompt: str) -> str:
        ...         return "качество"
        >>> client = MockLLMClient()
        >>> tags = run_pipeline_for_sku(
        ...     "data/raw/lamoda_reviews.csv",
        ...     "MP002XW0O0SI",
        ...     client,
        ...     name_to_tags={"Футболка": ["качество"]}
        ... )
        >>> len(tags) >= 0
        True
    """
    # Загружаем данные
    sku_data = load_dataset(csv_path=csv_path, min_reviews_per_sku=1)

    if sku not in sku_data:
        available_skus = list(sku_data.keys())[:10]
        raise ValueError(
            f"SKU '{sku}' не найден в файле. "
            f"Доступные SKU (первые 10): {available_skus}"
        )

    product_data = sku_data[sku]

    # Загружаем GOLDEN TAGS
    name_tags, subtype_tags, type_tags = load_golden_tags_from_dict(
        name_to_tags=name_to_tags,
        subtype_to_tags=subtype_to_tags,
        type_to_tags=type_to_tags,
    )

    # Запускаем инференс
    tags = run_inference(
        reviews=product_data["reviews"],
        llm_client=llm_client,
        name_to_tags=name_tags,
        subtype_to_tags=subtype_tags,
        type_to_tags=type_tags,
        product_name=product_data["name"],
        product_subtype=product_data["subtype"],
        product_type=product_data["type"],
        max_chars=max_chars,
        max_reviews=max_reviews,
        min_review_length=min_review_length,
        max_tags=max_tags,
        custom_prompt_template=custom_prompt_template,
    )

    return tags

