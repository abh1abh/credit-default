# Loan Default Prediction

A supervised machine learning project predicting borrower default on LendingClub personal loans. Built as preparation for an MSc in Data Science, with the explicit goal of developing modeling and honest-evaluation skills rather than shipping a polished app.

## Goal

Take a real, signal-rich tabular dataset and build a complete, rigorous default-prediction pipeline end to end — from raw data to a written report. The emphasis is on the parts that separate a data scientist from an engineer who can call `.fit()`: statistical reasoning, time-aware validation, avoiding data leakage, and not fooling myself during evaluation.

This is deliberately **not** a dashboard or web-app project. Engineering polish comes later, if at all — the summer is spent on modeling and analysis.

## Dataset

**LendingClub accepted loans, 2007–2018**
Kaggle: `wordsforthewise/lending-club` — https://www.kaggle.com/datasets/wordsforthewise/lending-club (CC0, public domain)

- ~2.2M loans, ~150 columns
- Includes origination-time FICO score ranges
- Tabular, finance-adjacent, contains genuine predictive signal

### Why this dataset

- **Tabular** — a clean bridge from prior Power BI / DAX work.
- **Real signal** — modeling choices visibly move the metric, so I can actually learn what feature engineering and model selection *do* (unlike market prediction, where nearly everything is noise).
- **A famous data-leakage trap** — the single best teaching artifact for the exact skill an MSc assumes and that engineers most often lack. Discovering and fixing it is the core of the project.

### Obtaining the data

1. Download the dataset zip from the Kaggle link above.
2. Extract `accepted_2007_to_2018Q4.csv.gz` into `data/raw/`.
3. Leave it gzipped — pandas reads gzip natively via `pd.read_csv(path)`.
4. Ignore `rejected_2007_to_2018Q4.csv.gz` — rejected loans were never funded and have no repayment outcome, so they're useless for default prediction.

Data is **never committed to git** (see Repo Structure). It's a static, public, ~600MB+ file — treated as a build artifact, re-fetched via these instructions.

## The core challenge: data leakage

LendingClub's raw file contains many columns that are only populated *after* a loan's outcome is known — payment totals, recoveries, last-payment dates, updated FICO. Throw everything into a model and you get a near-perfect AUC (~0.95+) that is completely fake: the model is reading the future.

The project's central task is building a **point-in-time feature set** — only what a lender knows *at origination*. Columns to remove include:

| Category | Columns |
|---|---|
| Payments & recoveries | `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `recoveries`, `collection_recovery_fee` |
| Last-payment / timing | `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`, `last_credit_pull_d` |
| Outstanding balance | `out_prncp`, `out_prncp_inv` |
| Updated FICO (post-origination) | `last_fico_range_high`, `last_fico_range_low` |
| Settlement / hardship | `debt_settlement_flag`, `settlement_*`, `hardship_*` |

Keep the **origination** `fico_range_high` / `fico_range_low` — those are legitimate. And wrestle with the subtle case: `verification_status` is arguably leaky, because LendingClub only verifies income on a subset of applications based on application content. Decide and justify.

After removing leakage, expect AUC to collapse to a realistic ~0.65–0.70.

## What makes this MSc-grade

Three things separate this from a typical Kaggle notebook:

1. **From-scratch math** — implementing logistic regression and cross-validation by hand before touching sklearn (Week 4).
2. **Time-aware validation + leakage discipline** — splits that respect loan issue dates; a rigorously point-in-time feature set (Weeks 2–3).
3. **A written report** — the actual deliverable, and the part nearly everyone skips (Week 9).

### The honest expectation

After leakage is removed, the model will *barely* beat the baseline. This is the real finding in essentially every serious attempt at this dataset — once leaky features are gone, there isn't much signal left. **That is not failure.** A report that says "naive setup gave 0.95 AUC, honest setup gave 0.68, and here's exactly why" demonstrates more maturity than a suspiciously brilliant result. Internalizing that is the point.

## Week-by-week plan

Assumes ~15–20 hrs/week over 9 weeks. Fewer hours → extend. The only non-negotiable week is Week 4.

### Week 0 — Setup + Python ramp (2–3 days)
- Install Python (`uv` or conda); set up Jupyter in VS Code.
- Speed-run pandas / numpy / matplotlib (one tutorial — this is fast, not a course).
- Initialize the git repo; download the dataset.
- **Deliverable:** notebook that loads the CSV and prints `.shape`, `.dtypes`, `.head()`, `.describe()`.

### Week 1 — Define the target + EDA
- Filter `loan_status` to "Fully Paid" vs "Charged Off" / "Default" → binary target. Drop "Current" loans (no outcome yet). *This filtering decision is itself a key conceptual task.*
- Compute the base rate (expect ~80–85% fully paid — this imbalance drives everything later).
- Build a missingness map; explore feature distributions.
- **Deliverable:** EDA notebook + a written "data understanding" section.
- **Watch:** the file is large. Load with `usecols=` and an explicit `dtype` map if memory is tight.

### Week 2 — The leakage trap (the centerpiece)
- Deliberately train a model on *all* columns; observe the absurd ~0.95+ AUC.
- Hunt down and remove every post-origination column (see leakage table above).
- Re-train; watch AUC drop to a realistic ~0.65–0.70.
- **Deliverable:** documented before/after with a one-line reason per dropped column.

### Week 3 — Train/test discipline + dumb baseline
- Carve out a held-out test set; don't touch it until Week 9.
- Make the split **time-aware** (train on earlier `issue_d`, test on later) — random splits leak temporal structure.
- Build baselines every real model must beat: majority-class, and single-feature (grade or FICO alone).
- **Deliverable:** locked test set + baseline metrics.

### Week 4 — Logistic regression from scratch (non-negotiable)
- Implement logistic regression with gradient descent in plain numpy.
- Write your own k-fold cross-validation loop.
- Compare against sklearn; confirm they match within tolerance.
- **Deliverable:** from-scratch implementation + a short writeup of the math (sigmoid, log-loss, gradient).

### Week 5 — Proper sklearn workflow + imbalance
- `Pipeline` + `ColumnTransformer` for encoding / scaling / imputation (no leakage across folds).
- Handle imbalance: compare `class_weight='balanced'` vs SMOTE — treat SMOTE skeptically and *test* whether it helps rather than assuming.
- Evaluate with the right metrics: ROC-AUC, PR-AUC, confusion matrix — **never accuracy** (an 85% base rate makes it a lie).
- **Deliverable:** clean pipeline + honest CV scores.

### Week 6 — Model family + honest tuning
- Train logistic regression, random forest, gradient boosting (LightGBM or XGBoost).
- Tune via validation split or nested CV — **zero peeking at test**.
- **Deliverable:** model comparison table.

### Week 7 — Cost-sensitive thresholds + calibration
- The default 0.5 threshold is wrong when errors cost differently (missed default = lost principal; rejected good borrower = lost margin). Build a cost matrix; find the cost-minimizing threshold.
- Check calibration (reliability curve, Brier score) — tree ensembles are often miscalibrated.
- **Deliverable:** threshold analysis tied to a concrete cost story.

### Week 8 — Unsupervised stretch + interpretation
- PCA for 2D visualization; k-means to segment borrowers.
- SHAP for feature importance; error analysis (who does the model fail on?).
- **Deliverable:** interpretation notebook.

### Week 9 — Final test + writeup
- Touch the held-out test set **once**; report final honest numbers.
- Write the report: question, data, method, results, limitations, next steps.
- Clean the repo + README for reproducibility.
- **Deliverable:** the report — the MSc-relevant artifact.

## Repo structure

```
loan-default-prediction/
├── data/
│   ├── raw/            # gitignored — see data/README.md
│   ├── interim/        # gitignored
│   └── processed/      # gitignored
├── notebooks/          # EDA, modeling, interpretation
├── src/                # reusable code (from-scratch LR, helpers)
├── reports/            # final writeup + figures
├── data/README.md      # how to obtain the dataset
├── .gitignore
└── README.md
```

`.gitignore` essentials:

```gitignore
data/raw/
data/processed/
data/interim/
__pycache__/
.ipynb_checkpoints/
*.pyc
.venv/
.env
```

**Everything data-shaped is a build artifact, not a commit.** Commit code and instructions; set `random_state` on every split so processed outputs are regenerable. Raw data exceeds GitHub's 100MB/file limit and is publicly re-downloadable anyway.

## Key principles

- **Never touch the test set until the end.** One look, Week 9.
- **Accuracy is a lie on imbalanced data.** Use ROC-AUC, PR-AUC, and cost-sensitive thresholds.
- **Splits respect time.** No random splits on data with an issue-date structure.
- **A point-in-time feature set is sacred.** If a lender wouldn't know it at origination, it doesn't go in the model.
- **Understand the math before the library.** `.fit()` is not a substitute for knowing what it does.
- **An honest null result beats a suspicious win.** Report what's real.
