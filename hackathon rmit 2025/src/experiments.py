"""Run comparable word, character, and combined TF-IDF experiments."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.validation import calculate_roc_auc, stratified_group_split


DEFAULT_CONFIGURATIONS = [
    {"feature_mode": feature_mode, "C": c}
    for c in (0.5, 2.0, 4.0)
    for feature_mode in ("word", "char", "combined")
]
LOG_COLUMNS = (
    "run_id",
    "experiment_id",
    "feature_mode",
    "C",
    "status",
    "roc_auc",
    "random_state",
    "validation_fraction_requested",
    "validation_fraction_actual",
    "train_rows",
    "validation_rows",
    "elapsed_seconds",
    "parameters",
    "error",
)


def _make_features(feature_mode: str) -> Any:
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True)
    if feature_mode == "word":
        return word
    if feature_mode == "char":
        return char
    if feature_mode == "combined":
        return FeatureUnion([("word", word), ("char", char)])
    raise ValueError(f"unsupported feature_mode: {feature_mode}")


def _clean_texts(texts: list[object]) -> list[str]:
    return ["" if pd.isna(text) else str(text) for text in texts]


def run_tfidf_experiments(
    texts: list[object],
    labels: list[object],
    *,
    validation_fraction: float = 0.2,
    random_state: int = 42,
    configurations: list[dict[str, Any]] | None = None,
    log_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run each configuration on the same leakage-aware split and optionally log CSV."""
    clean_texts = _clean_texts(list(texts))
    labels_array = np.asarray(labels)
    if len(clean_texts) != len(labels_array):
        raise ValueError("texts and labels must have the same length")
    train_indices, validation_indices = stratified_group_split(
        clean_texts,
        labels_array,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    actual_validation_fraction = len(validation_indices) / len(labels_array)
    configs = configurations if configurations is not None else DEFAULT_CONFIGURATIONS
    run_id = datetime.now(UTC).isoformat()
    records: list[dict[str, Any]] = []

    for index, configuration in enumerate(configs, start=1):
        feature_mode = str(configuration.get("feature_mode", "word"))
        c = float(configuration.get("C", 1.0))
        started = time.perf_counter()
        record: dict[str, Any] = {
            "run_id": run_id,
            "experiment_id": f"{index:02d}-{feature_mode}-C{c:g}",
            "feature_mode": feature_mode,
            "C": c,
            "status": "error",
            "roc_auc": np.nan,
            "random_state": random_state,
            "validation_fraction_requested": validation_fraction,
            "validation_fraction_actual": actual_validation_fraction,
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
            "elapsed_seconds": 0.0,
            "parameters": json.dumps(configuration, sort_keys=True),
            "error": "",
        }
        try:
            model = Pipeline(
                steps=[
                    ("features", _make_features(feature_mode)),
                    ("classifier", LogisticRegression(C=c, max_iter=2000, class_weight="balanced")),
                ]
            )
            model.fit([clean_texts[i] for i in train_indices], labels_array[train_indices])
            positive_index = list(model.classes_).index(1)
            scores = model.predict_proba([clean_texts[i] for i in validation_indices])[:, positive_index]
            record["roc_auc"] = calculate_roc_auc(labels_array[validation_indices], scores)
            record["status"] = "ok"
        except Exception as error:  # Keep other configurations running and preserve failure details.
            record["error"] = f"{type(error).__name__}: {error}"
        record["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        records.append(record)

    if log_path is not None:
        write_experiment_log(records, log_path)
    return records


def write_experiment_log(records: list[dict[str, Any]], output_path: str | Path) -> pd.DataFrame:
    """Write experiment rows to CSV with a stable column schema."""
    frame = pd.DataFrame(records, columns=LOG_COLUMNS)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare word, character, and combined TF-IDF models.")
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--text-column", required=True, help="Prompt text column name")
    parser.add_argument("--target-column", required=True, help="Binary label column name")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-csv", default="outputs/experiments.csv", help="Experiment results CSV path")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    train = pd.read_csv(args.train)
    required = (args.text_column, args.target_column)
    missing = [column for column in required if column not in train.columns]
    if missing:
        raise ValueError(f"train CSV is missing required column(s): {', '.join(missing)}")
    records = run_tfidf_experiments(
        train[args.text_column].tolist(),
        train[args.target_column].tolist(),
        validation_fraction=args.validation_fraction,
        random_state=args.seed,
        log_path=args.log_csv,
    )
    successful = [record for record in records if record["status"] == "ok"]
    summary = {
        "run_id": records[0]["run_id"] if records else None,
        "log_path": args.log_csv,
        "experiments": len(records),
        "successful_experiments": len(successful),
        "best": max(successful, key=lambda record: record["roc_auc"]) if successful else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
