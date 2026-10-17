import pandas as pd
import pytest

from src.data import audit_competition_data, load_competition_data


def write_competition_csvs(tmp_path, *, train=None, test=None, sample_submission=None):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"

    (train if train is not None else pd.DataFrame({"prompt": ["safe", "attack"], "label": [0, 1]})).to_csv(train_path, index=False)
    (test if test is not None else pd.DataFrame({"id": [10, 11], "prompt": ["one", "two"]})).to_csv(test_path, index=False)
    (sample_submission if sample_submission is not None else pd.DataFrame({"id": [10, 11], "target": [0.0, 0.0]})).to_csv(sample_path, index=False)

    return train_path, test_path, sample_path


def load_fixture(paths):
    return load_competition_data(
        *paths,
        text_column="prompt",
        target_column="label",
        id_column="id",
    )


def test_loads_csvs_and_audits_missing_values_classes_lengths_and_duplicates(tmp_path):
    paths = write_competition_csvs(
        tmp_path,
        train=pd.DataFrame({"prompt": ["safe", "attack", "attack", None], "label": [0, 1, 0, 1]}),
        test=pd.DataFrame({"id": [10, 11, 12], "prompt": ["test", None, "test"]}),
        sample_submission=pd.DataFrame({"id": [10, 11, 12], "target": [0.0, 0.0, 0.0]}),
    )

    train, test, sample_submission = load_fixture(paths)
    report = audit_competition_data(
        train,
        test,
        sample_submission,
        text_column="prompt",
        target_column="label",
        id_column="id",
    )

    assert report["row_counts"] == {"train": 4, "test": 3, "sample_submission": 3}
    assert report["class_counts"] == {0: 2, 1: 2}
    assert report["null_counts"]["train"]["prompt"] == 1
    assert report["text_length_stats"]["train"]["count"] == 3
    assert report["duplicate_text_rows"] == {"train": 1, "test": 1}


def test_rejects_missing_required_columns(tmp_path):
    paths = write_competition_csvs(
        tmp_path,
        train=pd.DataFrame({"body": ["safe"], "label": [0]}),
    )

    with pytest.raises(ValueError, match="train.*prompt"):
        load_fixture(paths)


def test_rejects_non_binary_or_missing_training_labels(tmp_path):
    paths = write_competition_csvs(
        tmp_path,
        train=pd.DataFrame({"prompt": ["safe", "attack"], "label": [0, 2]}),
    )

    with pytest.raises(ValueError, match="binary.*0.*1"):
        load_fixture(paths)


def test_rejects_duplicate_test_ids(tmp_path):
    paths = write_competition_csvs(
        tmp_path,
        test=pd.DataFrame({"id": [10, 10], "prompt": ["one", "two"]}),
        sample_submission=pd.DataFrame({"id": [10, 11], "target": [0.0, 0.0]}),
    )

    with pytest.raises(ValueError, match="test.*id.*unique"):
        load_fixture(paths)
