"""Create and validate competition submission CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def build_submission(
    test_ids: Sequence[object],
    scores: Sequence[float],
    sample_submission: pd.DataFrame,
    *,
    id_column: str = "id",
    target_column: str = "target",
) -> pd.DataFrame:
    """Return a validated submission matching the sample schema and ID order."""
    if sample_submission.columns.tolist() != [id_column, target_column]:
        raise ValueError(f"sample submission columns must be exactly [{id_column!r}, {target_column!r}] in that order")

    test_ids_list = list(test_ids)
    expected_ids = sample_submission[id_column].tolist()
    if len(test_ids_list) != len(expected_ids):
        raise ValueError("test IDs and sample submission row count must match")
    if test_ids_list != expected_ids:
        raise ValueError("test ID order must match the sample submission ID order")
    if len(set(test_ids_list)) != len(test_ids_list) or sample_submission[id_column].isna().any():
        raise ValueError("submission IDs must be unique and non-null")

    try:
        score_array = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("submission scores must be numeric") from error
    if score_array.ndim != 1 or len(score_array) != len(expected_ids):
        raise ValueError("submission scores and sample submission row count must match")
    if not np.isfinite(score_array).all():
        raise ValueError("submission scores must all be finite")
    if ((score_array < 0) | (score_array > 1)).any():
        raise ValueError("submission scores must be between 0 and 1")

    return pd.DataFrame({id_column: expected_ids, target_column: score_array})


def write_submission(
    test_ids: Sequence[object],
    scores: Sequence[float],
    sample_submission: pd.DataFrame,
    output_path: str | Path,
    *,
    id_column: str = "id",
    target_column: str = "target",
) -> pd.DataFrame:
    """Validate and write a Kaggle submission CSV without a pandas index."""
    submission = build_submission(
        test_ids,
        scores,
        sample_submission,
        id_column=id_column,
        target_column=target_column,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False)
    return submission
