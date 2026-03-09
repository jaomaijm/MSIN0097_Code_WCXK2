# Bank Marketing — Term Deposit Subscription Prediction

## Overview
This project builds an end-to-end predictive analytics pipeline to predict whether a bank client will subscribe to a term deposit, using the UCI Bank Marketing dataset. It was completed as part of **MSIN0097 Predictive Analytics** (UCL, 2025–26).

## Dataset
- **Source**: [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **File**: `bank-full.csv` (45,211 records, 17 features, semicolon-delimited)
- **Target**: `y` — whether the client subscribed to a term deposit (`yes` / `no`)
- For reproducibility, the dataset is loaded directly from the repository using a raw GitHub link rather than storing a local copy in the execution environment.
The dataset is loaded using:
```python
import pandas as pd
DATA_URL = "https://raw.githubusercontent.com/jaomaijm/MSIN0097_Code_WCXK2/main/bank-full.csv"
df = pd.read_csv(DATA_URL, sep=";")
## Project Structure
bank-marketing-project/
├── Bank_Marketing_Analysis.ipynb  # Main analysis notebook (90 cells, executed)
├── report.pdf                     # ~2000-word report with model card + appendix
├── build_notebook.py              # Notebook generation script
├── generate_report.py             # Report PDF generation script
├── requirements.txt               # Python dependencies (pip)
├── environment.yml                # Conda environment (Python 3.11)
├── README.md                      # This file
├── tests/
│   └── test_pipeline.py           # Standalone validation test suite
├── appendix/
│   └── agent_usage_log.md         # Agent Usage Log + Decision Register
└── fig_*.png                      # Generated figures (18 total)
```

## How to Run

### 1. Clone / Download the Repository
Download all files to a local directory.

### 2. Set Up the Environment

**Option A — Conda (recommended for reproducibility)**:
```bash
conda env create -f environment.yml
conda activate bank-marketing
```

**Option B — pip + venv**:
```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
jupyter notebook Bank_Marketing_Analysis.ipynb
```
Run all cells sequentially. The notebook is self-contained and will:
- Load and explore the data
- Preprocess features and split into train/validation/test sets
- Train and compare multiple models (Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Network)
- Fine-tune the best model via RandomizedSearchCV (60 iterations, 9 parameters, 5-fold CV)
- Perform error analysis, fairness audit, and poutcome sensitivity ablation
- Output final evaluation metrics and visualisations

### 4. Run Validation Tests
```bash
python tests/test_pipeline.py
```
The test suite validates data integrity, preprocessing correctness (including leakage detection), model sanity checks, and output file existence. All 45 test assertions should pass.

### 5. View the Report
Open `report.pdf` for the ~2,000-word analytical report including a model card and appendix.

## Requirements
- Python 3.11+ (pinned in `environment.yml`)
- See `requirements.txt` for all pip dependencies
- See `environment.yml` for full conda environment specification

## Reproducibility
- Random seeds are set throughout (`RANDOM_STATE = 42`) for reproducibility
- All preprocessing is implemented via scikit-learn ColumnTransformer pipelines
- Train/validation/test split ratios: 60/20/20 with stratified sampling
- Environment reproducible via `environment.yml` (conda) or `requirements.txt` (pip)
