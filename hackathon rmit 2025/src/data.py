"""Load and audit the competition CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _require_columns(frame: pd.DataFrame, frame_name: str, required: tuple[str, ...]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} CSV is missing required column(s): {', '.join(missing)}")


def _validate_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    *,
    text_column: str,
    target_column: str,
    id_column: str,
) -> None:
    _require_columns(train, "train", (text_column, target_column))
    _require_columns(test, "test", (id_column, text_column))
    _require_columns(sample_submission, "sample submission", (id_column, "target"))

    labels = train[target_column]
    observed_labels = set(labels.dropna().unique().tolist())
    if labels.isna().any() or observed_labels != {0, 1}:
        raise ValueError(
            f"train target labels must be binary with values 0 and 1 and contain both classes; "
            f"observed {sorted(observed_labels, key=str)}"
        )

    for frame_name, frame in (("test", test), ("sample submission", sample_submission)):
        ids = frame[id_column]
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError(f"{frame_name} ID column '{id_column}' must contain unique, non-null values")


def load_competition_data(
    train_path: str | Path,
    test_path: str | Path,
    sample_submission_path: str | Path,
    *,
    text_column: str,
    target_column: str,
    id_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three CSVs and fail early when their configured schema is invalid."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_submission_path)
    _validate_frames(
        train,
        test,
        sample_submission,
        text_column=text_column,
        target_column=target_column,
        id_column=id_column,
    )
    return train, test, sample_submission


def _text_length_summary(frame: pd.DataFrame, text_column: str) -> dict[str, int | float | None]:
    lengths = frame[text_column].dropna().astype("string").str.len()
    if lengths.empty:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "count": int(lengths.count()),
        "min": int(lengths.min()),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "max": int(lengths.max()),
    }


def audit_competition_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    *,
    text_column: str,
    target_column: str,
    id_column: str,
) -> dict[str, Any]:
    """Return row, null, label, text-length, and exact duplicate summaries."""
    _validate_frames(
        train,
        test,
        sample_submission,
        text_column=text_column,
        target_column=target_column,
        id_column=id_column,
    )
    frames = {"train": train, "test": test, "sample_submission": sample_submission}
    return {
        "row_counts": {name: int(len(frame)) for name, frame in frames.items()},
        "null_counts": {
            name: {str(column): int(count) for column, count in frame.isna().sum().items()}
            for name, frame in frames.items()
        },
        "class_counts": {
            label: int(count) for label, count in train[target_column].value_counts().sort_index().items()
        },
        "text_length_stats": {
            name: _text_length_summary(frame, text_column) for name, frame in (("train", train), ("test", test))
        },
        "duplicate_text_rows": {
            name: int(frame[text_column].dropna().duplicated().sum())
            for name, frame in (("train", train), ("test", test))
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load and audit the jailbreak competition CSV files.")
    parser.add_argument("--train", required=True, help="Path to the training CSV")
    parser.add_argument("--test", required=True, help="Path to the test CSV")
    parser.add_argument("--sample-submission", required=True, help="Path to the sample submission CSV")
    parser.add_argument("--text-column", required=True, help="Prompt text column name")
    parser.add_argument("--target-column", required=True, help="Training label column name")
    parser.add_argument("--id-column", required=True, help="Test and sample submission ID column name")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    frames = load_competition_data(
        args.train,
        args.test,
        args.sample_submission,
        text_column=args.text_column,
        target_column=args.target_column,
        id_column=args.id_column,
    )
    report = audit_competition_data(
        *frames,
        text_column=args.text_column,
        target_column=args.target_column,
        id_column=args.id_column,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
