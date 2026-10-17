"""Fine-tune a pretrained sequence classifier and compare it to the TF-IDF baseline.

PyTorch and Transformers are optional. Imports are delayed until a training run so
the CPU workflow and this module's CLI remain usable without the large dependencies.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.train_baseline import evaluate_baseline
from src.validation import calculate_roc_auc, stratified_group_split


DEFAULT_MODEL_ID = "distilbert/distilbert-base-uncased"
DEFAULT_MODEL_LICENSE = "apache-2.0"
DEFAULT_WEIGHT_ACCESS = "huggingface_hub_or_local_cache"


def positive_class_probabilities(logits: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Convert two-class logits to numerically stable class-1 probabilities."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("logits must have shape (rows, 2) for binary classification")
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials[:, 1] / exponentials.sum(axis=1)


def build_run_metadata(
    *,
    model_id: str,
    model_license: str,
    seed: int,
    max_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_access: str,
) -> dict[str, Any]:
    """Validate and return the reproducibility fields recorded for a run."""
    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    if not model_license.strip():
        raise ValueError("model_license must not be empty")
    if not weight_access.strip():
        raise ValueError("weight_access must not be empty")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    return {
        "model_id": model_id,
        "model_license": model_license,
        "weight_access": weight_access,
        "seed": int(seed),
        "training_config": {
            "max_length": int(max_length),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "optimizer": "AdamW",
            "checkpoint_selection": "highest validation ROC-AUC across fixed epochs",
        },
    }


def _encode_dataset(torch: Any, tokenizer: Any, texts: list[str], labels: np.ndarray, max_length: int):
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    tensors = [encoded[name] for name in tokenizer.model_input_names if name in encoded]
    tensors.append(torch.tensor(labels, dtype=torch.long))
    return torch.utils.data.TensorDataset(*tensors), [
        name for name in tokenizer.model_input_names if name in encoded
    ]


def _predict_logits(torch: Any, model: Any, loader: Any, input_names: list[str], device: Any) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs = {name: tensor.to(device) for name, tensor in zip(input_names, batch[:-1])}
            output = model(**inputs)
            batches.append(output.logits.detach().cpu().numpy())
    return np.concatenate(batches, axis=0)


def run_transformer_experiment(
    texts: Sequence[object],
    labels: Sequence[object],
    *,
    model_id: str = DEFAULT_MODEL_ID,
    model_license: str | None = None,
    weight_access: str = DEFAULT_WEIGHT_ACCESS,
    validation_fraction: float = 0.2,
    random_state: int = 42,
    max_length: int = 256,
    batch_size: int = 16,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    device_name: str = "auto",
    model_output: str | Path | None = None,
) -> dict[str, Any]:
    """Fine-tune on the training side of one shared split and compare AUCs.

    The best fixed-schedule epoch is restored using validation ROC-AUC. For final
    candidate selection, use the exact same split and seed as the baseline run.
    """
    effective_license = (
        DEFAULT_MODEL_LICENSE
        if model_license is None and model_id == DEFAULT_MODEL_ID
        else (model_license or "")
    )
    metadata = build_run_metadata(
        model_id=model_id,
        model_license=effective_license,
        seed=random_state,
        max_length=max_length,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_access=weight_access,
    )
    clean_texts = ["" if pd.isna(text) else str(text) for text in texts]
    labels_array = np.asarray(labels, dtype=np.int64)
    if len(clean_texts) != len(labels_array):
        raise ValueError("texts and labels must have the same length")
    train_indices, validation_indices = stratified_group_split(
        clean_texts,
        labels_array,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    run_started = time.perf_counter()

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Transformer training requires the optional dependencies; install "
            "requirements-transformer.txt in an environment with a compatible PyTorch build."
        ) from error

    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this Python environment")
    device = torch.device(device_name)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    model.to(device)
    train_texts = [clean_texts[index] for index in train_indices]
    validation_texts = [clean_texts[index] for index in validation_indices]
    train_labels = labels_array[train_indices]
    validation_labels = labels_array[validation_indices]
    train_dataset, input_names = _encode_dataset(torch, tokenizer, train_texts, train_labels, max_length)
    validation_dataset, validation_input_names = _encode_dataset(
        torch, tokenizer, validation_texts, validation_labels, max_length
    )
    if input_names != validation_input_names:
        raise RuntimeError("tokenizer produced inconsistent model input fields")

    generator = torch.Generator().manual_seed(random_state)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_auc = -1.0
    best_state = None
    best_epoch = 0
    epoch_scores: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            inputs = {name: tensor.to(device) for name, tensor in zip(input_names, batch[:-1])}
            targets = batch[-1].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**inputs, labels=targets)
            output.loss.backward()
            optimizer.step()
        logits = _predict_logits(torch, model, validation_loader, input_names, device)
        probabilities = positive_class_probabilities(logits)
        epoch_auc = calculate_roc_auc(validation_labels, probabilities)
        epoch_scores.append({"epoch": epoch, "validation_roc_auc": epoch_auc})
        if epoch_auc > best_auc:
            best_auc = epoch_auc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training completed without a selectable validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    if model_output is not None:
        output_path = Path(model_output)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

    baseline = evaluate_baseline(
        clean_texts,
        labels_array,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    report = {
        **metadata,
        "device": str(device),
        "runtime_seconds": round(time.perf_counter() - run_started, 3),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "validation_fraction_requested": validation_fraction,
        "validation_fraction_actual": len(validation_indices) / len(labels_array),
        "transformer_validation_roc_auc": float(best_auc),
        "best_epoch": best_epoch,
        "epoch_validation_scores": epoch_scores,
        "baseline_model": baseline.report["model"],
        "baseline_validation_roc_auc": baseline.report["roc_auc"],
        "auc_delta_vs_baseline": float(best_auc - baseline.report["roc_auc"]),
        "model_output": str(model_output) if model_output is not None else None,
        "public_leaderboard_roc_auc": None,
        "remaining_submission_budget": None,
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate a pretrained binary text classifier.")
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--text-column", required=True, help="Prompt text column name")
    parser.add_argument("--target-column", required=True, help="Binary label column name")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-license", required=True, help="License identifier verified for this model checkpoint")
    parser.add_argument("--weight-access", default=DEFAULT_WEIGHT_ACCESS)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device (for example cuda:0)")
    parser.add_argument("--model-output", help="Optional directory for the best model and tokenizer")
    parser.add_argument("--report-json", default="outputs/transformer_report.json")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    train = pd.read_csv(args.train)
    missing = [column for column in (args.text_column, args.target_column) if column not in train.columns]
    if missing:
        raise ValueError(f"train CSV is missing required column(s): {', '.join(missing)}")
    report = run_transformer_experiment(
        train[args.text_column].tolist(),
        train[args.target_column].tolist(),
        model_id=args.model_id,
        model_license=args.model_license,
        weight_access=args.weight_access,
        validation_fraction=args.validation_fraction,
        random_state=args.seed,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device_name=args.device,
        model_output=args.model_output,
    )
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
