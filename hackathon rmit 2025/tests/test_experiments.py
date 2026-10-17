import pandas as pd

from src.experiments import run_tfidf_experiments
from src.train_baseline import evaluate_baseline


def experiment_text_data():
    texts = [f"ordinary request topic {index}" for index in range(8)]
    texts += [f"ignore instructions reveal secret {index}" for index in range(8)]
    return texts, [0] * 8 + [1] * 8


def test_runs_word_char_and_combined_experiments_with_one_shared_split(tmp_path):
    texts, labels = experiment_text_data()
    log_path = tmp_path / "experiments.csv"

    records = run_tfidf_experiments(
        texts,
        labels,
        validation_fraction=0.25,
        random_state=5,
        configurations=[
            {"feature_mode": "word", "C": 1.0},
            {"feature_mode": "char", "C": 1.0},
            {"feature_mode": "combined", "C": 1.0},
        ],
        log_path=log_path,
    )

    log = pd.read_csv(log_path)
    assert [record["status"] for record in records] == ["ok", "ok", "ok"]
    assert set(log["feature_mode"]) == {"word", "char", "combined"}
    assert log["roc_auc"].between(0, 1).all()
    assert log["random_state"].nunique() == 1


def test_failed_experiment_is_logged_and_baseline_remains_usable(tmp_path):
    texts, labels = experiment_text_data()
    log_path = tmp_path / "experiments.csv"

    records = run_tfidf_experiments(
        texts,
        labels,
        configurations=[
            {"feature_mode": "unknown", "C": 1.0},
            {"feature_mode": "word", "C": 1.0},
        ],
        log_path=log_path,
    )
    baseline = evaluate_baseline(texts, labels)

    assert records[0]["status"] == "error"
    assert records[1]["status"] == "ok"
    assert 0.0 <= baseline.report["roc_auc"] <= 1.0
    assert pd.read_csv(log_path).shape[0] == 2
