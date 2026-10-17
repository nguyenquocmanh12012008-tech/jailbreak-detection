import numpy as np
import pytest

from src.validation import calculate_roc_auc, find_duplicate_text_overlap, stratified_group_split


def test_split_is_reproducible_stratified_and_keeps_exact_duplicates_together():
    texts = ["benign one", "benign two", "benign three", "attack one", "attack two", "attack three", "shared", "shared"]
    labels = [0, 0, 0, 1, 1, 1, 0, 1]

    first_train, first_validation = stratified_group_split(texts, labels, validation_fraction=0.25, random_state=17)
    second_train, second_validation = stratified_group_split(texts, labels, validation_fraction=0.25, random_state=17)

    assert np.array_equal(first_train, second_train)
    assert np.array_equal(first_validation, second_validation)
    assert set(np.asarray(labels)[first_validation]) == {0, 1}
    assert find_duplicate_text_overlap(np.asarray(texts)[first_train], np.asarray(texts)[first_validation]) == set()


def test_detects_duplicate_text_across_provided_splits():
    overlap = find_duplicate_text_overlap(["same", "train only"], ["same", "validation only"])

    assert overlap == {"same"}


def test_calculates_roc_auc_from_continuous_scores():
    score = calculate_roc_auc([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7])

    assert score == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([0, 1], [0.3], "same length"),
        ([1, 1], [0.3, 0.8], "both classes"),
        ([0, 1], [float("nan"), 0.8], "finite"),
    ],
)
def test_roc_auc_rejects_invalid_inputs(labels, scores, message):
    with pytest.raises(ValueError, match=message):
        calculate_roc_auc(labels, scores)
