"""Train and evaluate the classical TF-IDF baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.submission import write_submission
from src.validation import calculate_roc_auc, stratified_group_split


@dataclass
class BaselineEvaluation:
    pipeline: Pipeline
    validation_scores: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    report: dict[str, Any]


def build_baseline_pipeline(*, c: float = 4.0, min_df: int = 1) -> Pipeline:
    """Create the word n-gram TF-IDF and Logistic Regression baseline."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)),
            ("classifier", LogisticRegression(C=c, max_iter=2000, class_weight="balanced")),
        ]
    )


def _clean_texts(texts: list[object] | np.ndarray) -> list[str]:
    return ["" if pd.isna(text) else str(text) for text in texts]


def fit_full_baseline(
    texts: list[object] | np.ndarray,
    labels: list[object] | np.ndarray,
    *,
    c: float = 4.0,
    min_df: int = 1,
) -> Pipeline:
    """Fit the baseline pipeline on every labeled training row."""
    texts_clean = _clean_texts(list(texts))
    labels_array = np.asarray(labels)
    if len(texts_clean) != len(labels_array):
        raise ValueError("texts and labels must have the same length")
    if set(np.unique(labels_array).tolist()) != {0, 1}:
        raise ValueError("training labels must contain both classes 0 and 1")
    pipeline = build_baseline_pipeline(c=c, min_df=min_df)
    pipeline.fit(texts_clean, labels_array)
    return pipeline


def predict_positive_scores(pipeline: Pipeline, texts: list[object] | np.ndarray) -> np.ndarray:
    """Return class-1 probabilities for text inputs."""
    positive_class_index = list(pipeline.classes_).index(1)
    return pipeline.predict_proba(_clean_texts(list(texts)))[:, positive_class_index]


def evaluate_baseline(
    texts: list[object] | np.ndarray,
    labels: list[object] | np.ndarray,
    *,
    validation_fraction: float = 0.2,
    random_state: int = 42,
    c: float = 4.0,
    min_df: int = 1,
) -> BaselineEvaluation:
    """Fit on one leakage-aware split and report validation ROC-AUC."""
    texts_clean = _clean_texts(list(texts))
    labels_array = np.asarray(labels)
    if len(texts_clean) != len(labels_array):
        raise ValueError("texts and labels must have the same length")
    train_indices, validation_indices = stratified_group_split(
        texts_clean,
        labels_array,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )

    pipeline = fit_full_baseline(
        [texts_clean[index] for index in train_indices],
        labels_array[train_indices],
        c=c,
        min_df=min_df,
    )
    validation_scores = predict_positive_scores(
        pipeline, [texts_clean[index] for index in validation_indices]
    )
    auc = calculate_roc_auc(labels_array[validation_indices], validation_scores)
    report = {
        "model": "tfidf_word_ngrams_logistic_regression",
        "metric": "roc_auc",
        "roc_auc": auc,
        "random_state": random_state,
        "validation_fraction_requested": validation_fraction,
        "validation_fraction_actual": len(validation_indices) / len(labels_array),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "parameters": {"ngram_range": [1, 2], "min_df": min_df, "C": c, "class_weight": "balanced"},
    }
    return BaselineEvaluation(pipeline, validation_scores, train_indices, validation_indices, report)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a TF-IDF Logistic Regression baseline.")
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--text-column", required=True, help="Prompt text column name")
    parser.add_argument("--target-column", required=True, help="Binary label column name")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-json", help="Optional path for the JSON validation report")
    parser.add_argument("--test", help="Optional path to test CSV; enables submission generation")
    parser.add_argument("--sample-submission", help="Optional sample submission CSV")
    parser.add_argument("--output", help="Output CSV path for submission generation")
    parser.add_argument("--id-column", help="Test/sample submission ID column name")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    submission_arguments = (args.test, args.sample_submission, args.output, args.id_column)
    if any(submission_arguments) and not all(submission_arguments):
        parser.error("--test, --sample-submission, --output, and --id-column must be provided together")

    train = pd.read_csv(args.train)
    required = (args.text_column, args.target_column)
    missing = [column for column in required if column not in train.columns]
    if missing:
        raise ValueError(f"train CSV is missing required column(s): {', '.join(missing)}")
    result = evaluate_baseline(
        train[args.text_column].tolist(),
        train[args.target_column].tolist(),
        validation_fraction=args.validation_fraction,
        random_state=args.seed,
    )
    if all(submission_arguments):
        test = pd.read_csv(args.test)
        sample_submission = pd.read_csv(args.sample_submission)
        required_test_columns = (args.id_column, args.text_column)
        missing_test_columns = [column for column in required_test_columns if column not in test.columns]
        if missing_test_columns:
            raise ValueError(f"test CSV is missing required column(s): {', '.join(missing_test_columns)}")
        final_model = fit_full_baseline(train[args.text_column].tolist(), train[args.target_column].tolist())
        scores = predict_positive_scores(final_model, test[args.text_column].tolist())
        write_submission(
            test[args.id_column].tolist(),
            scores,
            sample_submission,
            args.output,
            id_column=args.id_column,
        )
        result.report["submission_path"] = str(args.output)
    output = json.dumps(result.report, indent=2)
    if args.report_json:
        Path(args.report_json).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
