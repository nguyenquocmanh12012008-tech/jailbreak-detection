# SPEC: RMIT Hackathon 2025 — Challenge 1 & 2

## Objective

Build a reproducible binary text-classification workflow that predicts whether a prompt is a jailbreak attempt. Challenge 1 is complete when the workflow creates and successfully submits its first valid Kaggle submission. Challenge 2 aims to maximize public leaderboard ROC-AUC, the competition metric, while using local validation to choose robust experiments. Challenge 1 is the onboarding milestone (20%); Challenge 2 is the main modeling challenge (50%).

The intended users are hackathon participants working in a Kaggle notebook or an equivalent local Python environment. The competition has ended, so this specification is also suitable for reproducing the workflow with the authorized competition data. The user has confirmed the Kaggle data mount is `/kaggle/input/competitions/rmit-hackathon-2025`, external pretrained models from basic through advanced may be used, and GPU compute is available in both intended environments.

## Assumptions

1. The training data contains one text field and a binary target, and the test data contains an identifier and the same text field. Exact column names must be confirmed from the competition files before implementation.
2. The target value `1` denotes jailbreak and `0` denotes benign; the submitted `target` value is a score/probability for class `1`.
3. The required submission columns are `id,target`, with one prediction per test row in the expected order.
4. Competition guidance, confirmed by the user, allows any tools, models, and frameworks from traditional ML through advanced LLM fine-tuning. This permission covers model choice; it does not by itself authorize adding external labeled training data.
5. Python 3.10 or newer and GPU compute are available in both Kaggle Notebook and the local environment. The baseline must still run without a GPU so the workflow has a dependable reference.
6. In Kaggle, the competition files are mounted at `/kaggle/input/competitions/rmit-hackathon-2025`; local data paths will be supplied through command-line arguments.

## Scope

### In scope

- Inspect train, test, and sample-submission files; validate schema, labels, missing values, class balance, text lengths, duplicates, and likely near-duplicates.
- Create a reproducible local validation split and report ROC-AUC.
- Establish a TF-IDF plus Logistic Regression baseline.
- Compare a small, documented set of text/vectorizer and model settings.
- Progress from classical ML through pretrained transformer/LLM fine-tuning as experiment results and runtime justify; compare candidates locally and use the public leaderboard to confirm promising submissions.
- Generate and validate a Kaggle-compatible CSV submission.
- Record experiment settings, validation scores, and output paths.

### Out of scope

- Challenge 3 game development and Challenge 4 adversarial prompt design.
- Any attempt to access private competition data without authorized Kaggle access.
- Collecting additional training data unless the competition rules explicitly allow it.
- Treating public leaderboard feedback as a substitute for local validation or assuming public scores guarantee private leaderboard performance.

## Success Criteria

1. A documented run can load the authorized train/test/sample-submission files without manual column edits after the schema is configured.
2. Data checks report row counts, required columns, nulls, target values, class balance, and exact duplicate counts; any schema or label violation stops the run with a clear error.
3. The baseline produces out-of-fold or held-out validation scores and ROC-AUC from a fixed seed and stratified split.
4. Validation has no exact duplicate prompt crossing from training to validation. Near-duplicate handling is documented when detected.
5. The final prediction file has exactly the required columns and test row count, contains finite scores in `[0,1]`, and preserves the sample-submission ID order.
6. At least one valid submission can be made for Challenge 1 when Kaggle access is available. For Challenge 2, each candidate is compared on the same local validation procedure and logged before spending a public submission attempt; public ROC-AUC is the optimization target.
7. The selected model and its preprocessing can be rerun from documented settings. The final attainable leaderboard score depends on the hidden test set and is not guaranteed.

## Tech Stack

- Python 3.10+
- pandas and NumPy for data handling
- scikit-learn for splitting, TF-IDF, Logistic Regression, and ROC-AUC
- SciPy sparse matrices through scikit-learn
- PyTorch and Hugging Face Transformers for GPU-based pretrained transformer experiments
- Kaggle Notebook and a local Python environment with GPU support and authorized data files

Pin versions in requirements files when the implementation is created. Keep transformer/GPU dependencies separate from the CPU baseline environment so the classical reference remains easy to install and run.

## Commands

Proposed baseline environment and execution commands (dataset paths and column names are configured after inspecting the actual files):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy scipy scikit-learn
python -m src.train_baseline --train data/train.csv --test data/test.csv --sample-submission data/sample_submission.csv --output submission.csv --seed 42
```

The `src.train_baseline` entry point is a proposed implementation contract, not an existing command. In Kaggle, use `/kaggle/input/competitions/rmit-hackathon-2025` for competition inputs and write `submission.csv` under `/kaggle/working/`; locally, pass the corresponding authorized file paths.

## Project Structure

```text
.
├── SPEC.md
├── requirements.txt
├── data/                         # Local authorized data only; do not commit
├── notebooks/
│   ├── 01_data_audit.ipynb
│   └── 02_baseline_and_experiments.ipynb
├── src/
│   ├── data.py                   # Schema checks and loading
│   ├── validation.py             # Reproducible split and metric
│   ├── train_baseline.py         # Baseline fit, validation, and predictions
│   └── submission.py             # Output checks and CSV writing
├── outputs/                      # Local experiment logs and model artifacts
└── submission.csv                # Generated artifact; do not commit by default
```

## Code Style

- Use descriptive `snake_case` names and small functions with explicit inputs and outputs.
- Keep data loading, validation, model fitting, and submission checks separate.
- Set and record all random seeds. Do not silently change labels, drop rows, or reorder test records.
- Keep preprocessing inside a scikit-learn `Pipeline` so validation and inference use the same transformations.
- Fail early with actionable messages when required columns, labels, IDs, or output checks do not match.

Example of the intended baseline pattern:

```python
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("classifier", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")),
])
pipeline.fit(train_text, train_target)
validation_score = pipeline.predict_proba(validation_text)[:, 1]
```

Treat these settings as an initial baseline to evaluate, not as known-optimal values. If the dataset is heavily imbalanced, compare settings using ROC-AUC and inspect class-specific error patterns rather than assuming `class_weight="balanced"` helps.

## Testing Strategy

- **Data checks:** required columns exist; IDs are unique where required; target values are binary; text nulls and duplicate rates are reported.
- **Validation checks:** stratified split is reproducible with the configured seed; ROC-AUC receives continuous positive-class scores, not thresholded labels.
- **Submission checks:** exact column names and order match the sample submission; row count and ID order match test; target scores are finite and within `[0,1]`.
- **Model checks:** compare the baseline and experiments on the same split. If near-duplicate or grouped samples are present, add a group-aware validation comparison to estimate leakage risk.
- A successful Kaggle submission is the end-to-end check for Challenge 1. It does not replace local validation for Challenge 2.

## Boundaries

- **Always:** follow the Kaggle competition rules; use only authorized data; preserve a validation set; record seeds and experiment settings; validate the submission file before uploading; keep credentials and private data out of version control.
- **Ask first:** adding external datasets, using a hosted API or paid compute, changing the label interpretation, or introducing a dependency with nontrivial runtime or licensing implications.
- **Never:** inspect or infer hidden test labels; leak validation labels into training; fabricate submission scores; commit Kaggle credentials or private competition data; use public leaderboard feedback as the only model-selection evidence.

## Experiment Plan

1. Inspect competition rules and actual file schema; replace the assumptions above with confirmed column names and constraints.
2. Make the first valid submission with a simple baseline to complete the Challenge 1 workflow.
3. Establish a fixed stratified local validation split and a TF-IDF Logistic Regression reference score.
4. Tune a small number of word/character n-gram and regularization settings; log every result.
5. Check duplicates and validation leakage; use group-aware validation if the data structure warrants it.
6. Progressively evaluate stronger pretrained transformer or LLM fine-tuning candidates on the available GPU when their local validation results and runtime justify the cost; record model, data, and runtime details.
7. Use public leaderboard submissions to confirm locally promising candidates, then select a model (or blend only when local validation supports it), retrain on all labeled training data, and validate the final submission format.

## Open Questions

1. What are the exact text, target, and identifier column names in the available competition files?
2. Is additional labeled training data permitted, or should training use only the provided competition data? Model/framework choice, including LLM fine-tuning, is confirmed as unrestricted.
3. Is network access available in the Kaggle runtime for downloading pretrained model weights, or must weights be attached/cached as Kaggle input?
4. Are Kaggle account access and submissions still available for this completed competition? Local submission validation does not depend on this.
