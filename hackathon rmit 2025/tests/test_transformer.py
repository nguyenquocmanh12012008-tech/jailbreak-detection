import numpy as np
import pytest

from src.train_transformer import build_run_metadata, positive_class_probabilities, run_transformer_experiment


def test_positive_class_probabilities_returns_stable_binary_softmax_scores():
    logits = np.array([[0.0, 0.0], [1000.0, 1001.0], [1001.0, 1000.0]])

    probabilities = positive_class_probabilities(logits)

    assert probabilities.shape == (3,)
    assert np.isfinite(probabilities).all()
    assert probabilities.tolist() == pytest.approx([0.5, 0.7310586, 0.2689414])


def test_run_metadata_records_reproducibility_and_weight_access():
    metadata = build_run_metadata(
        model_id="distilbert/distilbert-base-uncased",
        model_license="apache-2.0",
        seed=17,
        max_length=128,
        batch_size=8,
        epochs=2,
        learning_rate=3e-5,
        weight_access="huggingface_hub_or_local_cache",
    )

    assert metadata["model_id"] == "distilbert/distilbert-base-uncased"
    assert metadata["model_license"] == "apache-2.0"
    assert metadata["weight_access"] == "huggingface_hub_or_local_cache"
    assert metadata["seed"] == 17
    assert metadata["training_config"]["epochs"] == 2
    assert metadata["training_config"]["max_length"] == 128


def test_run_metadata_rejects_invalid_training_settings():
    with pytest.raises(ValueError, match="epochs"):
        build_run_metadata(
            model_id="example/model",
            model_license="apache-2.0",
            seed=1,
            max_length=128,
            batch_size=8,
            epochs=0,
            learning_rate=2e-5,
            weight_access="local",
        )


def test_custom_model_requires_explicit_license_metadata():
    texts = ["benign prompt", "jailbreak prompt", "another benign", "another jailbreak"]
    labels = [0, 1, 0, 1]

    with pytest.raises(ValueError, match="model_license"):
        run_transformer_experiment(texts, labels, model_id="some-org/custom-model")
