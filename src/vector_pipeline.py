# src/vector_pipeline.py
"""Pipeline для vectorize-режима: без LLM, только эмбеддинги и GOLDEN_TAGS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .data_loader import load_dataset, load_golden_tags_from_dict

logger = logging.getLogger(__name__)


# -----------------------------
# Вспомогательные структуры
# -----------------------------

@dataclass
class ProductMeta:
    sku: str
    name: Optional[str]
    subtype: Optional[str]
    type: Optional[str]
    reviews: List[str]
    num_reviews: int


@dataclass
class TagScore:
    tag: str
    score: float
    count: int


# -----------------------------
# Векторизатор тегов
# -----------------------------

class TagVectorizer:
    """
    Векторизатор тегов и отзывов.

    Логика:
      - эмбеддим все GOLDEN_TAGS один раз;
      - для продукта собираем кандидаты:
        name -> tags, subtype -> tags, type -> tags;
      - для отзывов делаем окна по 4 слова;
      - считаем cosine_similarity между окном и:
          * emb(TAG)
          * emb("не TAG")
        и голосуем за теги.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        window_size: int = 4,
        window_step: int = 1,
        # 🔥 более жесткий порог "похожести" окна на тег
        sim_threshold: float = 0.75,
        # 🔥 требуем заметный разрыв между TAG и "не TAG"
        neg_margin: float = 0.12,
        # 🔥 чуть строже дедуп
        dedup_threshold: float = 0.8,
        max_tags: int = 6,
        # 🔥 минимум положительных окон для тега
        min_tag_count: int = 3,
        min_review_length_words: int = 4,
        # 🔥 минимум разных отзывов, которые поддерживают тег
        min_reviews_with_evidence: int = 2,
        # 🔥 минимальный "сильный" максимум по тегу
        strong_max_threshold: float = 0.6,
        device: Optional[str] = None,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.window_size = window_size
        self.window_step = window_step
        self.sim_threshold = sim_threshold
        self.neg_margin = neg_margin
        self.dedup_threshold = dedup_threshold
        self.max_tags = max_tags
        self.min_tag_count = min_tag_count
        self.min_review_length_words = min_review_length_words
        self.min_reviews_with_evidence = min_reviews_with_evidence
        self.strong_max_threshold = strong_max_threshold

        # cache: tag -> (tag_emb, neg_tag_emb)
        self._tag_embedding_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # --------- служебные методы ---------

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        a: (N, D), b: (D,)
        return: (N,)
        """
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b_norm = b / (np.linalg.norm(b) + 1e-9)
        return np.dot(a_norm, b_norm)

    def _get_tag_embeddings(self, tag: str) -> Tuple[np.ndarray, np.ndarray]:
        if tag in self._tag_embedding_cache:
            return self._tag_embedding_cache[tag]

        texts = [tag, f"не {tag}"]
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        tag_emb, neg_emb = embs[0], embs[1]
        self._tag_embedding_cache[tag] = (tag_emb, neg_emb)
        return tag_emb, neg_emb

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # супер простой токенайзер — по пробелам, можно заменить на что-то лучше
        return [t for t in text.strip().split() if t]

    def _build_windows(self, review: str) -> List[str]:
        tokens = self._tokenize(review)
        if len(tokens) < self.min_review_length_words:
            return []

        windows: List[str] = []
        step = self.window_step
        size = self.window_size

        for i in range(0, max(1, len(tokens) - size + 1), step):
            window_tokens = tokens[i : i + size]
            if not window_tokens:
                continue
            windows.append(" ".join(window_tokens))

        # если вдруг ничего не получилось, добавим весь отзыв
        if not windows:
            windows = [" ".join(tokens)]

        return windows

    # --------- публичный метод для продукта ---------

    def infer_tags_for_product(
        self,
        reviews: List[str],
        candidate_tags: List[str],
        max_reviews: int = 50,
    ) -> List[str]:
        """
        Главная функция для одного SKU:
         - reviews: список текстов отзывов,
         - candidate_tags: список тегов-кандидатов (из GOLDEN_TAGS),
         - max_reviews: максимум отзывов, которые берём (для скорости).

        ⚠ Условия стали строже:
          - тег должен иметь достаточно окон с высокой похожестью,
          - эти окна должны принадлежать как минимум N разным отзывам,
          - максимум похожести по тегу должен быть >= strong_max_threshold,
          - "не TAG" не должен побеждать.
        """
        if not reviews or not candidate_tags:
            return []

        # обрежем количество отзывов для скорости
        if max_reviews is not None and len(reviews) > max_reviews:
            reviews = reviews[:max_reviews]

        # собираем окна по всем отзывам + индекс отзыва для каждого окна
        windows: List[str] = []
        window_review_idx: List[int] = []

        for review_idx, r in enumerate(reviews):
            r = str(r).strip()
            if not r:
                continue
            local_windows = self._build_windows(r)
            for w in local_windows:
                windows.append(w)
                window_review_idx.append(review_idx)

        if not windows:
            return []

        window_review_idx = np.array(window_review_idx, dtype=int)

        # эмбеддим все окна одним батчем
        window_embs = self.model.encode(
            windows,
            convert_to_numpy=True,
            show_progress_bar=False,
        )  # shape: (N_windows, D)

        tag_scores: List[TagScore] = []

        for tag in candidate_tags:
            tag = tag.strip()
            if not tag:
                continue

            tag_emb, neg_emb = self._get_tag_embeddings(tag)

            # cosineSimilarity(window, TAG) & cosineSimilarity(window, "не TAG")
            sim_tag = self._cosine_sim(window_embs, tag_emb)  # (N,)
            sim_neg = self._cosine_sim(window_embs, neg_emb)  # (N,)

            max_sim_tag = sim_tag.max()
            max_sim_neg = sim_neg.max()

            # 🔥 1) если максимум по "не TAG" выше, чем по TAG + запас — выкидываем тег
            if max_sim_neg > max_sim_tag + self.neg_margin:
                continue

            # 🔥 2) если сам максимум по TAG слабее, чем strong_max_threshold — выкидываем
            if max_sim_tag < self.strong_max_threshold:
                continue

            # окна, которые считаем свидетельством в пользу тега:
            # - sim_tag >= sim_threshold
            # - sim_tag > sim_neg + neg_margin
            mask_pos = (sim_tag >= self.sim_threshold) & (
                sim_tag > sim_neg + self.neg_margin
            )
            pos_indices = np.where(mask_pos)[0]

            if len(pos_indices) < self.min_tag_count:
                # 🔥 слишком мало сильных окон
                continue

            # 🔥 3) окна должны принадлежать как минимум N разным отзывам
            supported_reviews = np.unique(window_review_idx[pos_indices])
            if len(supported_reviews) < self.min_reviews_with_evidence:
                continue

            pos_scores = sim_tag[pos_indices]
            score = float(pos_scores.mean())
            count = int(len(pos_indices))
            tag_scores.append(TagScore(tag=tag, score=score, count=count))

        if not tag_scores:
            return []

        # сортируем теги по score (сначала сильные сигналы, потом по count)
        tag_scores.sort(key=lambda x: (x.score, x.count), reverse=True)

        # dedup похожих тегов по cosine similarity между эмбеддингами самих тегов
        selected_tags: List[str] = []
        selected_embs: List[np.ndarray] = []

        for ts in tag_scores:
            emb_tag, _ = self._get_tag_embeddings(ts.tag)

            if not selected_embs:
                selected_tags.append(ts.tag)
                selected_embs.append(emb_tag)
                continue

            sims = [
                self._cosine_sim(emb_tag.reshape(1, -1), emb_prev)[0]
                for emb_prev in selected_embs
            ]
            if max(sims) >= self.dedup_threshold:
                # слишком похож на уже выбранный тег — скипаем
                continue

            selected_tags.append(ts.tag)
            selected_embs.append(emb_tag)

            if len(selected_tags) >= self.max_tags:
                break

        return selected_tags


# -----------------------------
# Утилита: загрузка GOLDEN_TAGS
# -----------------------------

def load_golden_tags_from_json(
    golden_tags_path: Path,
) -> tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Загружает golden_tags.json и строит три словаря:
      name_to_tags, subtype_to_tags, type_to_tags.
    Формат ожидается примерно такой:
      [
        {"name": "Футболка", "tags": ["мягкий хлопок", ...]},
        {"subtype": "Кроссовки спортивные", "tags": [...]},
        {"type": "Обувь", "tags": [...]},
        ...
      ]
    """
    with open(golden_tags_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    name_to_tags: Dict[str, List[str]] = {}
    subtype_to_tags: Dict[str, List[str]] = {}
    type_to_tags: Dict[str, List[str]] = {}

    for item in data:
        tags = item.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        if not isinstance(tags, list):
            continue

        name = item.get("name")
        subtype = item.get("subtype") or item.get("good_subtype")
        product_type = item.get("type") or item.get("good_type")

        def add_to_dict(d: Dict[str, List[str]], key: Optional[str]):
            if not key:
                return
            if key not in d:
                d[key] = []
            d[key].extend(tags)

        add_to_dict(name_to_tags, name)
        add_to_dict(subtype_to_tags, subtype)
        add_to_dict(type_to_tags, product_type)

    # убираем дубли внутри каждого списка
    name_to_tags = {k: sorted(set(v)) for k, v in name_to_tags.items()}
    subtype_to_tags = {k: sorted(set(v)) for k, v in subtype_to_tags.items()}
    type_to_tags = {k: sorted(set(v)) for k, v in type_to_tags.items()}

    return name_to_tags, subtype_to_tags, type_to_tags


def get_candidate_tags_for_product(
    name: Optional[str],
    subtype: Optional[str],
    product_type: Optional[str],
    name_tags: Dict[str, List[str]],
    subtype_tags: Dict[str, List[str]],
    type_tags: Dict[str, List[str]],
) -> List[str]:
    """
    Собираем кандидаты тегов по правилу:
      - сначала по name,
      - если нет по name, то по subtype,
      - если нет по subtype, то по type.
    """
    candidates: List[str] = []

    if name and name in name_tags:
        candidates.extend(name_tags[name])

    if not candidates and subtype and subtype in subtype_tags:
        candidates.extend(subtype_tags[subtype])

    if not candidates and product_type and product_type in type_tags:
        candidates.extend(type_tags[product_type])

    # на всякий случай убираем дубли
    candidates = sorted(set(t.strip() for t in candidates if t.strip()))
    return candidates


# -----------------------------
# Pipeline для файла (vectorize)
# -----------------------------

def run_vector_pipeline_for_file(
    csv_path: str,
    golden_tags_path: str,
    output_path: Optional[str] = None,
    max_chars: int = 500,
    max_reviews: int = 50,
    min_review_length: int = 10,
    max_tags: int = 6,
    min_reviews_per_sku: int = 1,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Полный vectorize-pipeline:
      1. Загружаем CSV через load_dataset (по sku).
      2. Загружаем GOLDEN_TAGS и строим словари name/subtype/type -> tags.
      3. Для каждого SKU:
         - собираем кандидаты тегов,
         - прогоняем через TagVectorizer,
         - сохраняем результаты.
    """
    logger.info(f"[VECTOR] Начало обработки файла: {csv_path}")
    csv_path_obj = Path(csv_path)
    golden_tags_path_obj = Path(golden_tags_path)

    # 1. Загружаем данные по SKU
    sku_data_raw = load_dataset(
        csv_path=str(csv_path_obj),
        min_reviews_per_sku=min_reviews_per_sku,
    )
    logger.info(f"[VECTOR] Загружено {len(sku_data_raw)} SKU с отзывами")

    # 2. Загружаем GOLDEN_TAGS
    name_tags, subtype_tags, type_tags = load_golden_tags_from_json(golden_tags_path_obj)
    logger.info(
        f"[VECTOR] GOLDEN_TAGS: name={len(name_tags)}, "
        f"subtype={len(subtype_tags)}, type={len(type_tags)}"
    )

    # 3. Инициализируем TagVectorizer
    vectorizer = TagVectorizer(
        model_name=model_name,
        device=device,
        max_tags=max_tags,
    )

    # 4. Обработка всех SKU
    results: List[Dict[str, Any]] = []

    for i, (sku, pdata) in enumerate(sku_data_raw.items(), start=1):
        product = ProductMeta(
            sku=sku,
            name=pdata.get("name"),
            subtype=pdata.get("subtype"),
            type=pdata.get("type"),
            reviews=pdata.get("reviews", []),
            num_reviews=pdata.get("num_reviews", 0),
        )

        logger.info(
            f"[VECTOR] {i}/{len(sku_data_raw)} SKU={product.sku}, "
            f"name={product.name}, reviews={product.num_reviews}"
        )

        # фильтрация отзывов по длине и символам
        filtered_reviews = []
        for r in product.reviews:
            r = str(r).strip()
            if len(r) < min_review_length:
                continue
            if len(r) > max_chars:
                r = r[:max_chars]
            filtered_reviews.append(r)

        if not filtered_reviews:
            results.append(
                {
                    "sku": product.sku,
                    "name": product.name,
                    "subtype": product.subtype,
                    "type": product.type,
                    "tags": "",
                    "num_tags": 0,
                    "num_reviews": product.num_reviews,
                    "error": "no_valid_reviews",
                }
            )
            continue

        # кандидаты тегов
        candidate_tags = get_candidate_tags_for_product(
            name=product.name,
            subtype=product.subtype,
            product_type=product.type,
            name_tags=name_tags,
            subtype_tags=subtype_tags,
            type_tags=type_tags,
        )

        if not candidate_tags:
            results.append(
                {
                    "sku": product.sku,
                    "name": product.name,
                    "subtype": product.subtype,
                    "type": product.type,
                    "tags": "",
                    "num_tags": 0,
                    "num_reviews": product.num_reviews,
                    "error": "no_candidate_tags",
                }
            )
            continue

        # инференс тегов
        tags = vectorizer.infer_tags_for_product(
            reviews=filtered_reviews,
            candidate_tags=candidate_tags,
            max_reviews=max_reviews,
        )

        results.append(
            {
                "sku": product.sku,
                "name": product.name,
                "subtype": product.subtype,
                "type": product.type,
                "tags": ", ".join(tags),
                "num_tags": len(tags),
                "num_reviews": product.num_reviews,
                "error": None if tags else "no_selected_tags",
            }
        )

    df_results = pd.DataFrame(results)

    # 5. Сохранение результатов
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"[VECTOR] Результаты сохранены в: {out_path}")

    logger.info(
        f"[VECTOR] Готово: {len(df_results)} SKU, "
        f"{len(df_results[df_results['num_tags'] > 0])} с тегами"
    )

    return df_results

def run_vector_pipeline_for_sku(
    csv_path: str,
    golden_tags_path: str,
    sku: str,
    max_chars: int = 500,
    max_reviews: int = 50,
    min_review_length: int = 10,
    max_tags: int = 6,
    min_reviews_per_sku: int = 1,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device: Optional[str] = None,
) -> List[str]:
    """
    Vectorize-пайплайн для одной конкретной SKU.
    Возвращает список тегов (<= max_tags).
    """
    csv_path_obj = Path(csv_path)
    golden_tags_path_obj = Path(golden_tags_path)

    # 1. Загружаем все SKU
    sku_data_raw = load_dataset(
        csv_path=str(csv_path_obj),
        min_reviews_per_sku=min_reviews_per_sku,
    )

    if sku not in sku_data_raw:
        raise ValueError(f"SKU '{sku}' не найден в файле {csv_path}")

    pdata = sku_data_raw[sku]
    product = ProductMeta(
        sku=sku,
        name=pdata.get("name"),
        subtype=pdata.get("subtype"),
        type=pdata.get("type"),
        reviews=pdata.get("reviews", []),
        num_reviews=pdata.get("num_reviews", 0),
    )

    # 2. Загружаем GOLDEN_TAGS
    name_tags, subtype_tags, type_tags = load_golden_tags_from_json(golden_tags_path_obj)

    # 3. Фильтруем отзывы
    filtered_reviews: List[str] = []
    for r in product.reviews:
        r = str(r).strip()
        if len(r) < min_review_length:
            continue
        if len(r) > max_chars:
            r = r[:max_chars]
        filtered_reviews.append(r)

    if not filtered_reviews:
        return []

    # 4. Кандидаты тегов
    candidate_tags = get_candidate_tags_for_product(
        name=product.name,
        subtype=product.subtype,
        product_type=product.type,
        name_tags=name_tags,
        subtype_tags=subtype_tags,
        type_tags=type_tags,
    )

    if not candidate_tags:
        return []

    # 5. Векторизатор
    vectorizer = TagVectorizer(
        model_name=model_name,
        device=device,
        max_tags=max_tags,
    )

    tags = vectorizer.infer_tags_for_product(
        reviews=filtered_reviews,
        candidate_tags=candidate_tags,
        max_reviews=max_reviews,
    )

    # safety: на всякий случай обрежем ещё раз
    return tags[:max_tags]
