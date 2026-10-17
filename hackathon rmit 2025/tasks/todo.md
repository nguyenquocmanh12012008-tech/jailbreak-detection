# Task List: RMIT Hackathon 2025 — Challenge 1 & 2

Tasks are ordered by dependency. Commands assume execution from the repository root. The test harness and test dependencies are established in Task 1. Use synthetic fixtures so tests do not require private Kaggle files.

## Task 1: Set up Python project and test harness

**Description:** Create the minimal Python package/test setup for the modules specified in `SPEC.md`, pin or bound the baseline dependencies, and keep local data/artifacts out of version control.

**Acceptance criteria:**
- [x] Baseline and development/test dependencies are declared separately or clearly grouped.
- [x] `src` is importable as a Python module and pytest discovers tests.
- [x] Ignore rules exclude `.venv/`, local competition data, outputs, and generated submission files.

**Verification:**
- [x] `python -m pip install -r requirements.txt -r requirements-dev.txt` (run with `.venv\Scripts\python`)
- [x] `python -m pytest -q` — 4 passed (run with `.venv\Scripts\python`)

**Dependencies:** None

**Files likely touched:** `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `src/__init__.py`, `tests/test_project_setup.py`

**Estimated scope:** Small (5 files)

## Task 2: Implement data loading and schema audit

**Description:** Add CSV loading and configurable schema checks for train, test, and sample-submission data. Report row counts, nulls, class balance, text lengths, and exact duplicate counts; stop with actionable errors for invalid columns or labels.

**Acceptance criteria:**
- [x] Column names are passed/configured explicitly; no hidden assumptions about the actual dataset headers.
- [x] Train labels are binary and required text/ID columns are checked.
- [x] Audit results include missing values, class counts, text-length summary, and duplicate counts.
- [x] Unit tests use temporary synthetic CSVs and cover valid input plus missing columns/invalid labels.

**Verification:**
- [x] `python -m pytest -q tests/test_data.py` — 4 passed (run with `.venv\Scripts\python`)
- [x] `python -m pytest -q` — 8 passed (run with `.venv\Scripts\python`)
- [x] `python -m src.data --help` — audit CLI options displayed correctly (run with `.venv\Scripts\python`)

**Dependencies:** Task 1

**Files likely touched:** `src/data.py`, `tests/test_data.py`

**Estimated scope:** Small (2 files)

## Task 3: Implement reproducible validation and leakage checks

**Description:** Create the fixed-seed stratified train/validation split and ROC-AUC helper. Include exact duplicate overlap checks and a clear path to group-aware comparison if the audit identifies repeated prompts or groups.

**Acceptance criteria:**
- [x] Same seed yields identical split indices; both classes are represented in validation when feasible.
- [x] ROC-AUC uses continuous positive-class scores and rejects incompatible inputs with a clear error.
- [x] Exact duplicate overlap is detectable and the split keeps exact prompt groups together.
- [x] Unit tests cover determinism, class stratification, metric calculation, and duplicate leakage detection.

**Verification:**
- [x] `python -m pytest -q tests/test_validation.py` — 6 passed (run with `.venv\Scripts\python`)
- [x] `python -m pytest -q` — 14 passed (run with `.venv\Scripts\python`)

**Dependencies:** Task 1, Task 2

**Files likely touched:** `src/validation.py`, `tests/test_validation.py`

**Estimated scope:** Small (2 files)

### Checkpoint: After Tasks 1–3

- [x] `python -m pytest -q` passes (14 passed).
- [x] Data and validation modules can be exercised entirely with synthetic fixtures.
- [x] Split policy: exact matching prompts become groups and use stratified group folds, keeping duplicates on one side of the split.

## Task 4: Implement TF-IDF + Logistic Regression baseline

**Description:** Build a command-line baseline using a scikit-learn Pipeline, configurable text/target/ID columns, fixed validation settings, and a concise ROC-AUC/experiment report. Keep the baseline CPU-capable.

**Acceptance criteria:**
- [x] Vectorizer is fitted only on the training portion of each evaluation split through the scikit-learn Pipeline.
- [x] Baseline settings and seed are explicit and recorded in output.
- [x] Validation produces continuous class-1 scores and ROC-AUC.
- [x] Unit/integration test trains and scores on a small synthetic text dataset.

**Verification:**
- [x] `python -m pytest -q tests/test_baseline.py` — 2 passed (run with `.venv\Scripts\python`)
- [x] `python -m src.train_baseline --help` — CLI options displayed (run with `.venv\Scripts\python`)

**Dependencies:** Task 1, Task 2, Task 3

**Files likely touched:** `src/train_baseline.py`, `tests/test_baseline.py`

**Estimated scope:** Small (2 files)

## Task 5: Generate and validate Kaggle-format submission

**Description:** Retrain the selected baseline on all labeled training rows, predict test scores, preserve test/sample ID order, and write a CSV that matches the sample submission contract.

**Acceptance criteria:**
- [x] Output columns and order match the required `id,target` format/sample submission.
- [x] Output row count and ID order match test/sample submission.
- [x] Scores are finite and in `[0,1]`; invalid output is rejected before writing/upload.
- [x] Tests cover valid output and failures for wrong row count, IDs/order, NaN/inf, and out-of-range scores.

**Verification:**
- [x] `python -m pytest -q tests/test_submission.py` — 10 passed (run with `.venv\Scripts\python`)
- [x] `python -m pytest -q` — 26 passed, including synthetic CLI end-to-end (run with `.venv\Scripts\python`)
- [x] `python -m src.train_baseline --help` — CLI includes validation and submission options

**Dependencies:** Task 2, Task 4

**Files likely touched:** `src/submission.py`, `src/train_baseline.py`, `tests/test_submission.py`, `tests/test_baseline.py`

**Estimated scope:** Small (4 files)

### Checkpoint: After Tasks 4–5

- [ ] `python -m pytest -q` passes.
- [x] Synthetic end-to-end CLI run completes data audit, baseline validation, full-data fit, and submission CSV checks; real Kaggle files are not mounted here.
- [ ] If account access permits, upload the file manually to Kaggle and record whether the platform accepts it; otherwise record that external acceptance could not be checked.

## Task 6: Add controlled word/character TF-IDF experiments

**Description:** Compare a small predeclared set of word and character n-gram/regularization settings against the baseline using the same validation design, and save a machine-readable experiment table.

**Acceptance criteria:**
- [x] Every experiment uses the same split(s), metric, and seed as the baseline.
- [x] Word-only, char-only, and a documented combined/selected approach can be compared without using leaderboard score as the tuning metric.
- [x] Experiment settings, validation ROC-AUC, and run identifier are saved in a CSV/JSON log.
- [x] Test confirms experiment logging schema and that baseline remains available if an experiment fails.

**Verification:**
- [x] `python -m pytest -q tests/test_experiments.py` — 2 passed.
- [x] `python -m src.experiments --help` — CLI options displayed.
- [x] `python -m pytest -q` — 28 passed.

**Dependencies:** Task 3, Task 4

**Files likely touched:** `src/experiments.py`, `tests/test_experiments.py`, `src/train_baseline.py`

**Estimated scope:** Small (3 files)

## Task 7: Evaluate stronger pretrained transformer/LLM candidates on GPU

**Description:** Use the confirmed permission to choose any model/framework and available GPU to evaluate stronger pretrained transformer or LLM fine-tuning candidates against the saved classical baseline on the same validation design. Select model sizes and experiments according to validation evidence and runtime; public leaderboard ROC-AUC is the final competition objective, so use submissions to confirm promising candidates.

**Acceptance criteria:**
- [x] Pretrained model identifier/license, runtime, weight access method, and fine-tuning configuration are recorded for each run.
- [x] Tokenization does not fit on labels; model fitting uses only training-side rows, and validation labels are used only for ROC-AUC/checkpoint selection.
- [x] Best checkpoint is selected by validation metric over a fixed, documented epoch schedule.
- [x] Runtime, model identifier, seed, settings, and validation ROC-AUC are included in the JSON report.
- [x] The runner compares against the classical baseline using the same split and seed.
- [x] Public leaderboard result and remaining submission budget have separate report fields; they remain null until measured on Kaggle.

**Verification:**
- [x] `python -m pytest -q tests/test_transformer.py` — 4 passed; no model download required.
- [x] `python -m src.train_transformer --help` — CLI options displayed.
- [ ] Run a documented GPU smoke experiment for the selected model and compare against the baseline validation report.

**Environment limitation:** The Windows workspace does not have the Kaggle data mount or a detected GPU, and PyTorch/Transformers weights are not installed/downloaded here. The GPU smoke run and real-data model comparison must be run in the configured Kaggle GPU environment; no real-data or leaderboard metric is claimed.

**Dependencies:** Task 3, Task 4, Task 6

**Files likely touched:** `requirements-transformer.txt`, `src/train_transformer.py`, `tests/test_transformer.py`, `README.md`

**Estimated scope:** Small (4 files)

## Task 8: Document reproducible workflow and final review

**Description:** Document input paths/column configuration, environment setup, baseline and experiment commands, validation interpretation, submission checks, and the Kaggle manual upload step. Review the implementation against `SPEC.md` boundaries.

**Acceptance criteria:**
- [x] A new participant can follow the documented commands using authorized data after replacing the explicitly marked example filenames/columns with the inspected schema.
- [x] Documentation distinguishes local file validation from an actually accepted Kaggle submission.
- [x] Model identifier, weight source, and transformer comparison workflow are stated accurately; no measured transformer score is implied.
- [x] No competition data or credentials are present in the workspace; ignore rules exclude generated outputs and submissions.

**Verification:**
- [x] `python -m pytest -q` — 32 passed.
- [x] `python -m src.train_baseline --help` — CLI options displayed.
- [x] README commands documented; synthetic CLI smoke passed. The Kaggle data/GPU environment remains unverified here and is documented above.

**Dependencies:** Tasks 1–7

**Files likely touched:** `README.md`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`

**Estimated scope:** Small (4 files)
