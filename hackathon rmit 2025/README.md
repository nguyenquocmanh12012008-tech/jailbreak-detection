# RMIT Hackathon 2025 — Challenge 1 & 2

This repository contains a reproducible binary text-classification workflow for the jailbreak prompt challenge. It provides a schema audit, a leakage-aware TF-IDF baseline, controlled word/character experiments, optional pretrained transformer fine-tuning, and Kaggle submission validation.

For a self-contained notebook to upload or copy into Kaggle, start with [`kaggle_notebook.ipynb`](kaggle_notebook.ipynb). It does not require the local `src/` package to be mounted separately.

## Data and columns

In Kaggle, competition inputs are expected under `/kaggle/input/competitions/rmit-hackathon-2025`. Confirm the actual filenames and column names from that directory before running commands; this repository does not assume the competition headers. The examples below use `train.csv`, `test.csv`, `sample_submission.csv`, and placeholder columns `prompt`, `target`, and `id`. Replace those values with the confirmed schema.

The training target must contain both binary labels `0` and `1`; class `1` is scored as the positive class. The output submission follows the sample-submission ID order and writes class-1 probabilities.

## Install

The classical workflow needs Python 3.10+ and the packages listed in `requirements.txt` and `requirements-dev.txt`:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```

For transformer fine-tuning, install `requirements-transformer.txt` in an environment with a PyTorch build matching its CPU or CUDA runtime. Kaggle images may already have PyTorch installed. Check `torch.cuda.is_available()` in the notebook before selecting `--device cuda`.

## 1. Audit the data

```bash
python -m src.data \
  --train /kaggle/input/competitions/rmit-hackathon-2025/train.csv \
  --test /kaggle/input/competitions/rmit-hackathon-2025/test.csv \
  --sample-submission /kaggle/input/competitions/rmit-hackathon-2025/sample_submission.csv \
  --text-column prompt --target-column target --id-column id
```

Review row counts, null values, class counts, text lengths, and exact duplicate counts before training. The validation split keeps exact duplicate prompt text together. Near-duplicate text is not automatically grouped, so investigate it if the audit or domain suggests paraphrases may cross the split.

## 2. Train and create the baseline submission

This evaluates the baseline on a fixed-seed, stratified group split, then fits on all training rows and checks the generated submission:

```bash
python -m src.train_baseline \
  --train /kaggle/input/competitions/rmit-hackathon-2025/train.csv \
  --text-column prompt --target-column target \
  --validation-fraction 0.2 --seed 42 \
  --report-json /kaggle/working/baseline_report.json \
  --test /kaggle/input/competitions/rmit-hackathon-2025/test.csv \
  --sample-submission /kaggle/input/competitions/rmit-hackathon-2025/sample_submission.csv \
  --id-column id --output /kaggle/working/submission.csv
```

The baseline is word 1–2 gram TF-IDF with Logistic Regression. Its validation report is a local estimate, not a Kaggle leaderboard result. The submission writer validates required columns, row count, ID order, finite probabilities, and the `[0, 1]` score range. Upload the resulting file manually if the competition accepts submissions and record the platform response separately.

## 3. Compare TF-IDF experiments

```bash
python -m src.experiments \
  --train /kaggle/input/competitions/rmit-hackathon-2025/train.csv \
  --text-column prompt --target-column target \
  --validation-fraction 0.2 --seed 42 \
  --log-csv /kaggle/working/experiments.csv
```

This compares word, character, and combined n-gram features across the declared regularization settings. Every configuration uses the same split and ROC-AUC metric. Review `status`, `error`, settings, and validation AUC in the CSV; do not choose a model from public leaderboard feedback alone.

## 4. Optional GPU transformer comparison

The optional runner fine-tunes `distilbert/distilbert-base-uncased` for binary sequence classification by default. The model card lists the license as Apache-2.0 and describes the checkpoint as intended for downstream sequence-classification fine-tuning. Its weights can be loaded from the Hugging Face Hub or a local cache; make sure the runtime can access one of those locations. See the [model card](https://huggingface.co/distilbert/distilbert-base-uncased) and the [Transformers model loading API](https://huggingface.co/docs/transformers/main_classes/model).

```bash
python -m src.train_transformer \
  --train /kaggle/input/competitions/rmit-hackathon-2025/train.csv \
  --text-column prompt --target-column target \
  --model-id distilbert/distilbert-base-uncased \
  --model-license apache-2.0 --weight-access huggingface_hub_or_local_cache \
  --validation-fraction 0.2 --seed 42 \
  --device cuda --epochs 3 --batch-size 16 --max-length 256 \
  --model-output /kaggle/working/distilbert-jailbreak \
  --report-json /kaggle/working/transformer_report.json
```

The runner uses the same deterministic split policy and seed as the baseline, trains only on the training side, selects the best epoch by validation ROC-AUC, and records the baseline comparison, model/license, weight-access method, device, runtime, settings, and scores in JSON. The fixed epoch schedule is configurable; public leaderboard score and remaining submission budget are intentionally recorded as unknown until measured on Kaggle.

## Validation and competition results

The local ROC-AUC is measured on a held-out, exact-duplicate-grouped split. It is useful for comparing experiments under the same procedure, but it is not the public leaderboard ROC-AUC and does not guarantee hidden-test performance. Do not report a public score until Kaggle returns one. This repository has not run against the mounted competition data or a GPU in the local Windows workspace; run the commands above in the authorized Kaggle environment to produce those measurements.

## Project files

- `src/data.py` — CSV schema checks and data audit.
- `src/validation.py` — deterministic duplicate-grouped validation and ROC-AUC.
- `src/train_baseline.py` — baseline evaluation, final fit, and submission creation.
- `src/experiments.py` — controlled word/character TF-IDF experiments.
- `src/train_transformer.py` — optional pretrained sequence-classifier fine-tuning.
- `src/submission.py` — Kaggle submission contract checks.
- `SPEC.md`, `tasks/plan.md`, `tasks/todo.md` — approved scope and implementation plan.

Local datasets, experiment outputs, virtual environments, and generated submission files are excluded by `.gitignore`.
