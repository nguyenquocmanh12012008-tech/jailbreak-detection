# Implementation Plan: RMIT Hackathon 2025 — Challenge 1 & 2

## Overview

Implement the approved `SPEC.md` as a reproducible Python workflow for jailbreak prompt classification. The core path starts with authorized input/schema checks, establishes leakage-aware local ROC-AUC validation, fits a TF-IDF + Logistic Regression baseline, and emits a validated Kaggle-format submission. Challenge 2 experiments build on that fixed baseline. Kaggle submission itself is a manual, account-dependent step because the competition has ended.

## Architecture Decisions

- Keep data loading/schema checks, validation, model training, and submission formatting in separate `src/` modules, as described by `SPEC.md`.
- Use scikit-learn Pipelines so text preprocessing is fitted only on training folds and reused consistently for validation and inference.
- Use `/kaggle/input/competitions/rmit-hackathon-2025` as the confirmed Kaggle data root and make local/Kaggle paths and text, target, and ID column names configurable. The files are not mounted in this Windows workspace, so inspect them in Kaggle before fixing the column settings.
- Use a fixed seed and stratified split as the initial validation design. If audit finds duplicates/groups that make this split optimistic, add a group-aware comparison before selecting a model.
- Progress from CPU-capable classical baselines to GPU pretrained transformer/LLM fine-tuning as local validation and runtime justify. Competition guidance allows any model/framework; keep GPU dependencies separate so the baseline remains independently runnable.
- Keep competition data, credentials, and generated artifacts out of version control.

## Dependency Graph

```text
1. Environment and test harness
        ↓
2. Data loading and schema audit
        ↓
3. Validation split and metrics
        ↓
4. Baseline training and validation
        ↓
5. Submission generation and end-to-end checks
        ↓
6. Controlled TF-IDF experiments
        ↓
7. GPU pretrained transformer/LLM comparison
        ↓
8. Reproducible run documentation / final review
```

## Task List

### Phase 1: Foundation and data contract

- [x] Task 1: Set up Python project and test harness
- [x] Task 2: Implement data loading and schema audit
- [x] Task 3: Implement reproducible validation and leakage checks

### Checkpoint: Data foundation

- [x] `python -m pytest -q` passes
- [x] Synthetic fixtures cover valid and invalid schemas without requiring private competition files
- [x] Validation split and metric are repeatable for the configured seed

### Phase 2: Baseline submission path

- [x] Task 4: Implement TF-IDF + Logistic Regression baseline and validation report
- [x] Task 5: Generate and validate Kaggle-format submission

### Checkpoint: End-to-end baseline

- [x] A local command reads authorized inputs, trains the baseline, reports ROC-AUC, and creates a valid submission CSV (verified with synthetic CSV fixtures)
- [x] Output IDs/order and scores pass automated checks
- [ ] If Kaggle access is available, manually upload once and confirm Kaggle accepts the file; this completes the Challenge 1 submission milestone

### Phase 3: Challenge 2 improvement and reproducibility

- [x] Task 6: Add controlled word/character TF-IDF experiments and experiment log
- [ ] Task 7: Implement optional pretrained transformer/LLM fine-tuning and run the GPU comparison (runner and unit tests complete; GPU run pending)
- [x] Task 8: Document the reproducible workflow and review boundaries

### Checkpoint: Challenge 1 & 2 ready for review

- [x] All automated tests pass (32 passed).
- [ ] Run transformer-versus-baseline model comparison on authorized competition data and GPU; local smoke data is synthetic only.
- [x] Final commands, settings, metric, and output format are documented.
- [x] No competition data or credentials are present; ignore rules exclude local data and generated outputs. Workspace has no Git metadata, so tracked-file status could not be inspected.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Kaggle-mounted data is unavailable from the local workspace or access changes | High | Build tests with synthetic fixtures; use the confirmed Kaggle mount for real-data runs and only report scores actually measured |
| Actual column names or labels differ from assumptions | High | Inspect sample/train/test schema first; make names explicit CLI arguments and fail clearly |
| Exact or near-duplicate prompts cross validation split | High | Report duplicates during audit; keep exact duplicates together or remove them from validation comparison; test group-aware validation when justified |
| Public leaderboard overfitting or limited submission attempts | Medium | Log local validation results first; submit promising candidates and track public ROC-AUC changes |
| Pretrained weights cannot be downloaded in the Kaggle runtime | Medium | Check network access early; attach/cache weights as input when allowed; keep the CPU baseline runnable and report the transformer blocker accurately |
| Competition has ended, preventing Kaggle upload | Medium | Complete local file validation and document the manual Kaggle step as unavailable; don't treat local file checks as an actual accepted Kaggle submission |

## Open Questions

- Confirm the exact input column names and target encoding by inspecting the files at the Kaggle mount.
- Clarify whether additional labeled training data is allowed; any model/framework choice, including LLM fine-tuning, and GPU availability in both Kaggle and local environments are confirmed.
- Check whether Kaggle Notebook network access permits downloading model weights or whether weights must be attached as input.
- Kaggle account access is required to perform the Challenge 1 submission milestone; the plan does not assume account access.

## Task List Target

Detailed checklists, acceptance criteria, files, dependencies, and verification commands are recorded in [`tasks/todo.md`](todo.md). No external tracker is designated.
