"""Модуль для загрузки и подготовки данных из CSV."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def load_dataset(
    csv_path: str,
    review_column: str = "comment_text",
    sku_column: str = "product_sku",
    name_column: str = "name",
    subtype_column: str = "good_subtype",
    type_column: str = "good_type",
    min_reviews_per_sku: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    Загружает датасет из CSV и группирует отзывы по SKU.

    Args:
        csv_path: Путь к CSV файлу с отзывами.
        review_column: Название колонки с текстом отзывов. По умолчанию "comment_text".
        sku_column: Название колонки с SKU продукта. По умолчанию "product_sku".
        name_column: Название колонки с названием продукта. По умолчанию "name".
        subtype_column: Название колонки с подтипом продукта. По умолчанию "good_subtype".
        type_column: Название колонки с типом продукта. По умолчанию "good_type".
        min_reviews_per_sku: Минимальное количество отзывов для включения SKU. По умолчанию 1.

    Returns:
        Словарь вида {sku: {reviews: List[str], name: Optional[str], 
        subtype: Optional[str], type: Optional[str]}}.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если обязательные колонки отсутствуют в CSV.

    Examples:
        >>> data = load_dataset("data/raw/lamoda_reviews.csv")
        >>> len(data) > 0
        True
        >>> "reviews" in list(data.values())[0]
        True
    """
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        raise FileNotFoundError(f"Файл не найден: {csv_path}")

    # Загружаем CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Ошибка при чтении CSV файла: {e}") from e

    # Проверяем наличие обязательных колонок
    required_columns = [sku_column, review_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Отсутствуют обязательные колонки в CSV: {missing_columns}. "
            f"Доступные колонки: {list(df.columns)}"
        )

    # Удаляем строки с пустыми отзывами
    df = df.dropna(subset=[review_column])
    df = df[df[review_column].astype(str).str.strip() != ""]

    if df.empty:
        raise ValueError("CSV файл не содержит валидных отзывов")

    # Группируем по SKU
    sku_data: Dict[str, Dict[str, Any]] = {}

    for sku, group in df.groupby(sku_column):
        # Получаем отзывы для данного SKU
        reviews = group[review_column].dropna().astype(str).tolist()
        reviews = [r.strip() for r in reviews if r.strip()]

        # Фильтруем по минимальному количеству отзывов
        if len(reviews) < min_reviews_per_sku:
            continue

        # Получаем метаданные продукта (берем первое значение, так как они должны быть одинаковыми)
        name = None
        if name_column in group.columns:
            name_values = group[name_column].dropna().unique()
            name = name_values[0] if len(name_values) > 0 else None

        subtype = None
        if subtype_column in group.columns:
            subtype_values = group[subtype_column].dropna().unique()
            subtype = subtype_values[0] if len(subtype_values) > 0 else None

        product_type = None
        if type_column in group.columns:
            type_values = group[type_column].dropna().unique()
            product_type = type_values[0] if len(type_values) > 0 else None

        sku_data[sku] = {
            "reviews": reviews,
            "name": name,
            "subtype": subtype,
            "type": product_type,
            "num_reviews": len(reviews),
        }

    return sku_data


def load_golden_tags_from_dict(
    name_to_tags: Optional[Dict[str, List[str]]] = None,
    subtype_to_tags: Optional[Dict[str, List[str]]] = None,
    type_to_tags: Optional[Dict[str, List[str]]] = None,
) -> tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Загружает словари GOLDEN TAGS из переданных аргументов.

    Args:
        name_to_tags: Словарь связки name -> список тегов.
        subtype_to_tags: Словарь связки subtype -> список тегов.
        type_to_tags: Словарь связки type -> список тегов.

    Returns:
        Кортеж из трех словарей (name_to_tags, subtype_to_tags, type_to_tags).

    Examples:
        >>> name_tags = {"Футболка": ["качество", "размер"]}
        >>> n, s, t = load_golden_tags_from_dict(name_to_tags=name_tags)
        >>> "Футболка" in n
        True
    """
    return (
        name_to_tags or {},
        subtype_to_tags or {},
        type_to_tags or {},
    )

