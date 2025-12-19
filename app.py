# app.py
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
import numpy as np

from src.vector_pipeline import (
    TagVectorizer,
    load_golden_tags_from_json,
    get_candidate_tags_for_product,
)
from src.data_loader import load_dataset


# ------------------------
# Константы / настройки
# ------------------------

DEFAULT_CSV_PATH = "/Users/macbook/bootcamp-lamoda/data/lamoda_reviews.csv"
DEFAULT_GOLDEN_PATH = "data/golden_tags_2_cleaned.json"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MAX_TAGS = 6
MAX_CHARS = 500
MAX_REVIEWS = 50
MIN_REVIEW_LENGTH = 50


# ------------------------
# Кэши
# ------------------------

@st.cache_data(show_spinner=False)
def cached_load_dataset(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Загрузка и группировка отзывов по SKU (кэшируется)."""
    return load_dataset(csv_path=csv_path, min_reviews_per_sku=1)


@st.cache_data(show_spinner=False)
def cached_load_golden_tags(golden_path: str):
    """Загрузка GOLDEN_TAGS (кэшируется)."""
    return load_golden_tags_from_json(Path(golden_path))


@st.cache_resource(show_spinner=False)
def cached_vectorizer(model_name: str) -> TagVectorizer:
    """Инициализация и кэширование эмбеддинговой модели."""
    return TagVectorizer(
        model_name=model_name,
        max_tags=MAX_TAGS,
    )


# ------------------------
# Логика для одной SKU
# ------------------------

def get_tags_and_evidence_for_sku(
    sku: str,
    csv_path: str,
    golden_tags_path: str,
    max_chars: int = MAX_CHARS,
    max_reviews: int = MAX_REVIEWS,
    min_review_length: int = MIN_REVIEW_LENGTH,
    max_tags: int = MAX_TAGS,
    top_reviews_per_tag: int = 5,
) -> Tuple[List[str], Dict[str, List[Tuple[str, float]]], Dict[str, Any]]:
    """
    Возвращает:
      - список тегов,
      - словарь tag -> [(review, score), ...],
      - словарь с метаданными продукта.
    """
    # 1) Данные по всем SKU
    sku_data = cached_load_dataset(csv_path)
    if sku not in sku_data:
        raise ValueError(f"SKU '{sku}' не найден в CSV")

    pdata = sku_data[sku]
    product_meta = {
        "sku": sku,
        "name": pdata.get("name"),
        "subtype": pdata.get("subtype"),
        "type": pdata.get("type"),
        "num_reviews": pdata.get("num_reviews", 0),
    }

    # 2) Фильтрация отзывов
    raw_reviews: List[str] = pdata.get("reviews", [])
    filtered_reviews: List[str] = []
    for r in raw_reviews:
        r = str(r).strip()
        if len(r) < min_review_length:
            continue
        if len(r) > max_chars:
            r = r[:max_chars]
        filtered_reviews.append(r)

    if not filtered_reviews:
        return [], {}, product_meta

    if len(filtered_reviews) > max_reviews:
        filtered_reviews = filtered_reviews[:max_reviews]

    # 3) GOLDEN_TAGS
    name_tags, subtype_tags, type_tags = cached_load_golden_tags(golden_tags_path)

    candidate_tags = get_candidate_tags_for_product(
        name=product_meta["name"],
        subtype=product_meta["subtype"],
        product_type=product_meta["type"],
        name_tags=name_tags,
        subtype_tags=subtype_tags,
        type_tags=type_tags,
    )

    if not candidate_tags:
        return [], {}, product_meta

    # 4) Векторизатор
    vectorizer = cached_vectorizer(MODEL_NAME)

    # Основной инференс тегов (окна по 4 слова и т.п.)
    tags = vectorizer.infer_tags_for_product(
        reviews=filtered_reviews,
        candidate_tags=candidate_tags,
        max_reviews=max_reviews,
    )

    tags = tags[:max_tags]

    # 5) Подбор "объясняющих" отзывов для каждого тега
    #    Здесь проще: считаем симилярность между эмбеддингом тега и эмбеддингами ПОЛНЫХ отзывов.
    if not tags:
        return [], {}, product_meta

    review_embs = vectorizer.model.encode(
        filtered_reviews,
        convert_to_numpy=True,
        show_progress_bar=False,
    )  # (N_reviews, D)

    evidence: Dict[str, List[Tuple[str, float]]] = {}

    for tag in tags:
        tag_emb, _ = vectorizer._get_tag_embeddings(tag)  # используем уже обученный кеш
        sims = vectorizer._cosine_sim(review_embs, tag_emb)  # (N_reviews,)

        # берем топ-N отзывов по симилярности
        top_idx = np.argsort(-sims)[:top_reviews_per_tag]
        tag_evidence: List[Tuple[str, float]] = []
        for idx in top_idx:
            tag_evidence.append((filtered_reviews[idx], float(sims[idx])))

        evidence[tag] = tag_evidence

    return tags, evidence, product_meta


# ------------------------
# UI
# ------------------------

def main():
    st.set_page_config(
        page_title="Lamoda Tags Demo (vectorize)",
        page_icon="👟",
        layout="wide",
    )

    st.title("Lamoda SKU → теги (vectorize) 👟")
    st.markdown(
        """
Это демо без LLM: теги подбираются по эмбеддингам и GOLDEN_TAGS.

**Как пользоваться:**
1. Укажи путь к CSV с отзывами и к файлу golden_tags.json  
2. Введи SKU  
3. Нажми "Получить теги"
        """
    )

    # --- Сайдбар с настройками ---
    st.sidebar.header("Настройки")

    csv_path = st.sidebar.text_input(
        "Путь к CSV с отзывами",
        value=DEFAULT_CSV_PATH,
    )
    golden_path = st.sidebar.text_input(
        "Путь к GOLDEN_TAGS JSON",
        value=DEFAULT_GOLDEN_PATH,
    )
    max_chars = st.sidebar.number_input(
        "Максимальная длина отзыва (символы)",
        min_value=50,
        max_value=2000,
        value=MAX_CHARS,
        step=50,
    )
    max_reviews = st.sidebar.number_input(
        "Максимум отзывов на SKU",
        min_value=5,
        max_value=200,
        value=MAX_REVIEWS,
        step=5,
    )
    min_review_length = st.sidebar.number_input(
        "Мин. длина отзыва (символы)",
        min_value=50,
        max_value=200,
        value=MIN_REVIEW_LENGTH,
        step=5,
    )
    top_reviews_per_tag = st.sidebar.number_input(
        "Сколько отзывов показывать на тег",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Модель эмбеддингов: " + MODEL_NAME)

    # --- Основной ввод SKU ---
    st.subheader("Ввод SKU")
    sku = st.text_input(
        "Введите SKU товара",
        value="MP002XW0FXPS",
        placeholder="Например: MP002XW0FXPS",
    )

    run_btn = st.button("Получить теги")

    if run_btn:
        if not sku.strip():
            st.warning("Введите SKU.")
            return

        # Нормализуем пути
        csv_path_abs = str(Path(csv_path).expanduser())
        golden_path_abs = str(Path(golden_path).expanduser())

        with st.spinner("Считаем эмбеддинги и ищем теги..."):
            try:
                tags, evidence, meta = get_tags_and_evidence_for_sku(
                    sku=sku.strip(),
                    csv_path=csv_path_abs,
                    golden_tags_path=golden_path_abs,
                    max_chars=max_chars,
                    max_reviews=max_reviews,
                    min_review_length=min_review_length,
                    max_tags=MAX_TAGS,
                    top_reviews_per_tag=top_reviews_per_tag,
                )
            except Exception as e:
                st.error(f"Ошибка: {e}")
                st.exception(e)
                st.text(traceback.format_exc())
                return

        if not tags:
            st.info("Теги не найдены (нет отзывов / нет кандидатов / мало сигналов).")
            return

        # --- Вывод общей инфы по продукту ---
        st.markdown("### Информация о товаре")
        cols = st.columns(4)
        cols[0].metric("SKU", meta.get("sku", "—"))
        cols[1].metric("Название", meta.get("name", "—"))
        cols[2].metric("Подтип", meta.get("subtype", "—"))
        cols[3].metric("Тип", meta.get("type", "—"))

        st.markdown(f"**Всего отзывов по SKU:** {meta.get('num_reviews', '—')}")

        # --- Теги ---
        st.markdown("### Теги")
        st.write(", ".join(tags))

        st.markdown("### Отзывы по тегам")

        for tag in tags:
            st.markdown(f"#### 🏷️ {tag}")
            tag_reviews = evidence.get(tag, [])
            if not tag_reviews:
                st.write("_Нет отзывов для этого тега_")
                continue

            for i, (text, score) in enumerate(tag_reviews, start=1):
                with st.expander(f"Отзыв {i} (similarity={score:.3f})", expanded=(i == 1)):
                    st.write(text)


if __name__ == "__main__":
    main()