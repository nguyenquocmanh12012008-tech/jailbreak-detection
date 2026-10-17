# 🛡️ RMIT Hackathon 2025 — Jailbreak Prompt Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![ROC-AUC](https://img.shields.io/badge/Private%20ROC--AUC-0.97901-brightgreen)](#-competition-results)

An end-to-end, reproducible Natural Language Processing (NLP) pipeline designed for the **RMIT Hackathon 2025** challenge. This project develops an intelligent binary text classifier capable of detecting adversarial "jailbreak" prompts targeting Large Language Models (LLMs) versus benign user interactions.

---

## 🏆 Competition Results

Our submission achieved top-tier performance on the competitive Kaggle leaderboard:

- 🥇 **Private Score (ROC-AUC): `0.97901`**
- 🥈 **Public Score (ROC-AUC): `0.97500`**

<div align="center">
  <img src="assets/kaggle_submission_results.png" alt="RMIT Hackathon 2025 Kaggle Submission Results" width="900" />
</div>

<br>

### 📊 Submission Breakdown

| Submission / Model Version | Description | Public Score (ROC-AUC) | Private Score (ROC-AUC) | Status |
| :--- | :--- | :---: | :---: | :---: |
| **`qmanh hackathon rmit2025 - Version 1`** | **Ensemble / Fine-Tuned Transformer Pipeline** 🏆 | **0.97500** | **0.97901** | **Best Score** |
| `notebookb6b4016d03 - Version 1` | Fine-tuned Transformer Checkpoint | 0.96136 | 0.97362 | Completed |
| `submission.csv` | Full-fit Multi-gram Baseline | 0.96136 | 0.97362 | Completed |
| `submission_transformer.csv` | Specialized Sequence Classifier (`Jailbreak-Detector`) | 0.96751 | 0.97266 | Completed |
| `submission (2).csv` | Calibrated Transformer Probabilities | 0.96751 | 0.97266 | Completed |

---

## 🎯 Problem Statement & Overview

As Large Language Models (LLMs) are increasingly integrated into mission-critical systems, securing them against adversarial exploitation is paramount. **Jailbreak attacks** craft sophisticated prompts using social engineering, obfuscation, persona adoption, or roleplay to circumvent safety alignment guardrails.

This competition tasks participants with developing a high-precision binary text classification model:
- **Class 0 (`benign`)**: Normal, safe user queries.
- **Class 1 (`jailbreak`)**: Adversarial prompts attempting to bypass safety policies.
- **Evaluation Metric**: Area Under the Receiver Operating Characteristic Curve (**ROC-AUC**).

---

## 💡 Key Architectural Highlights & Methodology

### 1. 🛡️ Leakage-Free Validation (`StratifiedGroupKFold`)
Prompt datasets frequently contain exact duplicates or near-duplicate paraphrases. Random splitting introduces severe data leakage, inflating local cross-validation scores. 
- We enforced an **exact-duplicate grouping strategy** with `StratifiedGroupKFold`.
- All duplicate instances of a prompt are strictly contained within either the train or validation fold, guaranteeing zero data leakage and a cross-validation score that closely mirrors the hidden private test set.

### 2. 🔠 Hybrid Multi-Granularity Classical ML Pipeline
- **Word n-grams (1–2)**: Captures semantic intent and key trigger terms with sublinear term-frequency scaling (`sublinear_tf=True`).
- **Character n-grams (2–5) with boundary awareness (`char_wb`)**: Critical for detecting evasion tricks, adversarial misspellings, leetspeak, obfuscated tokens, and unusual punctuation syntax.
- **Classifiers**: Balanced `LogisticRegression` and `LinearSVC` with systematic grid-search over regularization parameters ($C \in \{0.5, 2.0, 4.0\}$).

### 3. 🤖 Deep Learning & Transformer Fine-Tuning
- **Domain-Specialized Backbone**: Leveraged `madhurjindal/Jailbreak-Detector` and `distilbert-base-uncased` sequence classification backbones.
- **Dynamic Batch Padding**: `DataCollatorWithPadding` dynamically pads batches to the longest prompt instead of a static maximum length, significantly boosting GPU throughput.
- **Training Optimization**:
  - Mixed precision (`fp16`) for acceleration.
  - Gradient accumulation to maintain optimal effective batch sizes.
  - `EarlyStoppingCallback` monitored on validation ROC-AUC to prevent overfitting.
- **Probability Calibration**: Softmax temperature scaling ensuring well-calibrated confidence probabilities within $[0, 1]$.

---

## 📁 Repository Structure

```text
├── assets/
│   └── kaggle_submission_results.png   # Kaggle leaderboard submission screenshot
├── src/
│   ├── __init__.py
│   ├── data.py                         # Data audit, schema validation, & stats
│   ├── validation.py                   # Leakage-aware StratifiedGroupKFold & metrics
│   ├── train_baseline.py               # TF-IDF baseline training & submission pipeline
│   ├── experiments.py                  # Grid search for word/char n-grams & classifiers
│   ├── train_transformer.py            # Pretrained Transformer fine-tuning pipeline
│   └── submission.py                   # Kaggle submission contract verification
├── tests/                              # Automated test suite (32 unit tests passing)
│   ├── test_baseline.py
│   ├── test_data.py
│   ├── test_experiments.py
│   ├── test_project_setup.py
│   ├── test_submission.py
│   ├── test_transformer.py
│   └── test_validation.py
├── kaggle_notebook.ipynb               # Self-contained Kaggle notebook (zero local dependencies)
├── kaggle_train.py                     # Standalone CLI runner for Kaggle GPU environments
├── requirements.txt                    # Core runtime dependencies
├── requirements-dev.txt                # Development & testing tools (pytest)
├── requirements-transformer.txt        # PyTorch & Hugging Face dependencies
├── SPEC.md                             # Architectural specification
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone this repository and set up a virtual environment:

```bash
# Clone repository
git clone https://github.com/<your-username>/rmit-hackathon-2025.git
cd rmit-hackathon-2025

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt -r requirements-dev.txt
```

*(Optional) For Transformer GPU training:*
```bash
python -m pip install -r requirements-transformer.txt
```

### 2. Run Test Suite

Verify all components and integrity constraints:

```bash
python -m pytest -q
# Output: 32 passed
```

### 3. Step-by-Step Pipeline Execution

#### Step 1: Audit Dataset & Check for Leakage
```bash
python -m src.data \
  --train path/to/train.csv \
  --test path/to/test.csv \
  --sample-submission path/to/sample_submission.csv \
  --text-column prompt \
  --target-column target \
  --id-column id
```

#### Step 2: Run TF-IDF & Linear Experiments
```bash
python -m src.experiments \
  --train path/to/train.csv \
  --text-column prompt \
  --target-column target \
  --validation-fraction 0.2 \
  --seed 42 \
  --log-csv outputs/experiments.csv
```

#### Step 3: Train Transformer with GPU Acceleration
```bash
python -m src.train_transformer \
  --train path/to/train.csv \
  --text-column prompt \
  --target-column target \
  --model-id madhurjindal/Jailbreak-Detector \
  --device cuda \
  --epochs 5 \
  --batch-size 16 \
  --max-length 512 \
  --model-output outputs/jailbreak-model \
  --report-json outputs/transformer_report.json
```

---

## ⚡ Running on Kaggle

This project provides two self-contained solutions ready to run out of the box in Kaggle kernels:

1. **Interactive Notebook**: Open [`kaggle_notebook.ipynb`](kaggle_notebook.ipynb) in Kaggle, select GPU accelerator (P100 / T4), and run all cells.
2. **Kaggle CLI Script**:
   ```bash
   # Run classical TF-IDF baseline:
   python kaggle_train.py

   # Fine-tune GPU Transformer:
   python kaggle_train.py --transformer --epochs 6 --model-id madhurjindal/Jailbreak-Detector
   ```

Both options automatically generate a verified, format-compliant `submission.csv` in `/kaggle/working/`.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Machine Learning:** Scikit-Learn, SciPy, NumPy, Pandas
- **Deep Learning / NLP:** PyTorch, Hugging Face `transformers`, `tokenizers`, `accelerate`
- **Quality Assurance:** Pytest, Flake8, Strict Type Hints

---

## 📜 License & Acknowledgements

- Built for the **RMIT Hackathon 2025**.
- Model weights: `madhurjindal/Jailbreak-Detector` (MIT License) and `distilbert-base-uncased` (Apache 2.0).
- Open source under the [MIT License](LICENSE).

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Machine Learning:** Scikit-Learn, SciPy, NumPy, Pandas
- **Deep Learning / NLP:** PyTorch, Hugging Face `transformers`, `tokenizers`, `accelerate`
- **Quality Assurance:** Pytest, Flake8, Strict Type Hints

---

## 📜 License & Acknowledgements

- Built for the **RMIT Hackathon 2025**.
- Model weights: `madhurjindal/Jailbreak-Detector` (MIT License) and `distilbert-base-uncased` (Apache 2.0).
- Open source under the [MIT License](LICENSE).

