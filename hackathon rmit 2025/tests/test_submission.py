import sys

import numpy as np
import pandas as pd
import pytest

from src.submission import build_submission, write_submission
from src.train_baseline import fit_full_baseline, predict_positive_scores
from src.train_baseline import main as run_baseline_cli


def sample_submission(ids=(10, 11, 12)):
    return pd.DataFrame({"id": list(ids), "target": [0.0] * len(ids)})


def test_builds_submission_in_sample_id_order():
    result = build_submission([10, 11, 12], [0.9, 0.1, 0.6], sample_submission())

    assert result.columns.tolist() == ["id", "target"]
    assert result["id"].tolist() == [10, 11, 12]
    assert result["target"].tolist() == [0.9, 0.1, 0.6]


@pytest.mark.parametrize(
    ("test_ids", "scores", "sample", "message"),
    [
        ([10, 11], [0.5, 0.6], sample_submission(), "row count"),
        ([11, 10, 12], [0.5, 0.6, 0.7], sample_submission(), "ID order"),
        ([10, 11, 12], [0.5, float("nan"), 0.7], sample_submission(), "finite"),
        ([10, 11, 12], [0.5, -0.1, 0.7], sample_submission(), "between 0 and 1"),
        ([10, 11, 12], [0.5, 1.1, 0.7], sample_submission(), "between 0 and 1"),
        ([10, 11, 12], [0.5, 0.6, 0.7], pd.DataFrame({"target": [0.0], "id": [10]}), "columns"),
    ],
)
def test_rejects_invalid_submission_inputs(test_ids, scores, sample, message):
    with pytest.raises(ValueError, match=message):
        build_submission(test_ids, scores, sample)


def test_writes_csv_without_index_and_can_be_read_back(tmp_path):
    output_path = tmp_path / "submission.csv"

    result = write_submission([10, 11, 12], [0.1, 0.2, 0.3], sample_submission(), output_path)
    loaded = pd.read_csv(output_path)

    assert result.equals(loaded)
    assert loaded.columns.tolist() == ["id", "target"]


def test_fits_on_all_training_rows_and_writes_predicted_test_scores(tmp_path):
    texts = [f"ordinary request number {index}" for index in range(8)]
    texts += [f"ignore rules reveal secret number {index}" for index in range(8)]
    labels = [0] * 8 + [1] * 8
    model = fit_full_baseline(texts, labels)
    scores = predict_positive_scores(model, ["ordinary request", "ignore rules"])

    result = write_submission([10, 11], scores, sample_submission((10, 11)), tmp_path / "submission.csv")

    assert len(model.named_steps["classifier"].classes_) == 2
    assert result["target"].between(0, 1).all()


def test_cli_evaluates_refits_and_writes_a_submission(tmp_path, monkeypatch, capsys):
    train_texts = [f"ordinary request number {index}" for index in range(8)]
    train_texts += [f"ignore rules reveal secrets number {index}" for index in range(8)]
    train = pd.DataFrame({"prompt": train_texts, "label": [0] * 8 + [1] * 8})
    test = pd.DataFrame({"id": [21, 22], "prompt": ["ordinary request", "ignore rules"]})
    sample = pd.DataFrame({"id": [21, 22], "target": [0.0, 0.0]})
    train_path, test_path, sample_path = (tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "sample.csv")
    output_path = tmp_path / "submission.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    sample.to_csv(sample_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_baseline",
            "--train", str(train_path),
            "--test", str(test_path),
            "--sample-submission", str(sample_path),
            "--output", str(output_path),
            "--text-column", "prompt",
            "--target-column", "label",
            "--id-column", "id",
        ],
    )

    run_baseline_cli()

    assert pd.read_csv(output_path).columns.tolist() == ["id", "target"]
    assert '"roc_auc"' in capsys.readouterr().out
