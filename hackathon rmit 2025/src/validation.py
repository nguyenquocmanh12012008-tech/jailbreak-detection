"""Leakage-aware validation splits and ROC-AUC helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def _text_groups(texts: Sequence[object]) -> np.ndarray:
    group_by_text: dict[tuple[str, object], int] = {}
    groups: list[int] = []
    for index, text in enumerate(texts):
        key = ("missing", index) if pd.isna(text) else ("text", str(text))
        if key not in group_by_text:
            group_by_text[key] = len(group_by_text)
        groups.append(group_by_text[key])
    return np.asarray(groups)


def stratified_group_split(
    texts: Sequence[object],
    labels: Sequence[object],
    *,
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/validation row indices, keeping exact prompt duplicates together."""
    texts_array = np.asarray(texts, dtype=object)
    labels_array = np.asarray(labels)
    if texts_array.ndim != 1 or labels_array.ndim != 1 or len(texts_array) != len(labels_array):
        raise ValueError("texts and labels must be one-dimensional arrays of the same length")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if set(np.unique(labels_array).tolist()) != {0, 1}:
        raise ValueError("labels must contain both classes 0 and 1")

    class_counts = np.unique(labels_array, return_counts=True)[1]
    desired_splits = max(2, round(1 / validation_fraction))
    n_splits = min(desired_splits, int(class_counts.min()), len(np.unique(_text_groups(texts_array))))
    if n_splits < 2:
        raise ValueError("not enough examples and distinct prompt groups to make a stratified split")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    groups = _text_groups(texts_array)
    overall_positive_rate = float(labels_array.mean())
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for train_indices, validation_indices in splitter.split(np.zeros(len(labels_array)), labels_array, groups):
        validation_labels = labels_array[validation_indices]
        if len(np.unique(validation_labels)) < 2:
            continue
        size_error = abs(len(validation_indices) / len(labels_array) - validation_fraction)
        balance_error = abs(float(validation_labels.mean()) - overall_positive_rate)
        candidates.append((size_error + balance_error, train_indices, validation_indices))

    if not candidates:
        raise ValueError("could not create a validation fold containing both classes; reduce validation_fraction")
    _, train_indices, validation_indices = min(candidates, key=lambda candidate: candidate[0])
    return train_indices, validation_indices


def find_duplicate_text_overlap(
    train_texts: Sequence[object], validation_texts: Sequence[object]
) -> set[str]:
    """Return non-null exact prompt strings found in both provided splits."""
    train_values = {str(text) for text in train_texts if not pd.isna(text)}
    validation_values = {str(text) for text in validation_texts if not pd.isna(text)}
    return train_values.intersection(validation_values)


def calculate_roc_auc(labels: Sequence[object], scores: Sequence[float]) -> float:
    """Compute binary ROC-AUC from finite positive-class probability or decision scores."""
    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores, dtype=float)
    if labels_array.ndim != 1 or scores_array.ndim != 1 or len(labels_array) != len(scores_array):
        raise ValueError("labels and scores must be one-dimensional arrays of the same length")
    if set(np.unique(labels_array).tolist()) != {0, 1}:
        raise ValueError("ROC-AUC validation requires both classes 0 and 1")
    if not np.isfinite(scores_array).all():
        raise ValueError("ROC-AUC scores must all be finite")
    return float(roc_auc_score(labels_array, scores_array))
