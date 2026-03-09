# Plan: EDA-First Modeling Pipeline

Start by deepening EDA to understand each table, timestamp fields, and join keys, then define a target after the data reveals what is feasible. Use a resident-week frame with a 30-day horizon, strictly pre-incident features, and a time-based split for validation. Build a minimal, traceable pipeline in notebooks plus lightweight helpers under src, and capture findings and decisions in the README.

## Steps
1. Expand EDA in [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb): column summaries, missingness, timestamp ranges, and join key coverage across tables; flag candidate event timestamps and potential leakage fields.
2. Profile the datavision dataset in [notebooks/02_datavision_eda.ipynb](notebooks/02_datavision_eda.ipynb) and flag sparse or low-signal features.
3. Maintain the data dictionary in [README.md](README.md): table purposes, key columns, time columns, and known caveats.
4. Confirm targets based on prevalence and business relevance; document decisions in [README.md](README.md).
5. Build a resident-week dataset with rolling lookback windows using only pre-incident data; keep feature assembly helpers in [src/data](src/data) and [src/features](src/features).
6. Train three models per label (rule-based, decision tree, LightGBM) with a time-based split; run notebooks under [notebooks](notebooks).
7. Tune LightGBM with Optuna, capture feature importance, and summarize results in [README.md](README.md).

## Verification
- Re-run the notebook(s) end-to-end; confirm no leakage by checking feature timestamps vs label window.
- Validate label prevalence and split stability (train vs validation rates, AUROC/PR-AUC, calibration).

## Decisions
- EDA first to choose a feasible, defensible target.
- Snapshot dataset every 7 days, 30-day label horizon.
- Strictly pre-incident features only.
- Time-based train/validation split.
- Active resident definition: admission_date <= snapshot <= discharge_date (or null discharge).
- Snapshot range (current): 2023-08-01 to 2025-01-31.
- Return-to-hospital label uses hospital_admissions only and excludes planned admissions when status is available; transfers are features only.
- Modeling labels in notebooks use fall_next_30d and rth_next_30d (mapped from label_fall_30d/label_rth_30d if needed).
- Missing values are handled per split (days_since_last_* -> 365, other numeric -> train medians).
- Feature stability filtering uses train + validation only; test drift is logged only.
- Decision trees use max_depth=4 and permutation importance on validation data.
- LightGBM tuning uses Optuna (TPE sampler + MedianPruner) and scale_pos_weight for imbalance.

# Agent Instructions --- Tricura ML Case

## Objective

Build a machine learning system to **predict the risk of incidents for
residents in skilled nursing facilities** using the Tricura dataset.

The goal is to help insurance clients **reduce incident frequency and
severity** by identifying residents at elevated risk.

The solution must emphasize:

-   Clear ML problem framing
-   Temporal data handling (avoid leakage)
-   Feature engineering across multiple medical datasets
-   Interpretable models
-   Business impact

------------------------------------------------------------------------

# Business Context

Tricura Insurance provides liability insurance to skilled nursing
facilities.

Incidents involving residents generate insurance claims.

  Incident Type        \% Claims   Avg Cost
  -------------------- ----------- ----------
  Falls                13%         \$3,500
  Medication errors    10%         \$5,000
  Wounds               7%          \$4,000
  Return-to-hospital   7%          \$20,000
  Elopement            5%          \$2,500
  Altercations         2%          \$2,500

Increasing premiums is not sustainable. The strategic lever is
**reducing incidents**.

The ML system should allow facilities to **identify high-risk residents
early and intervene**.

------------------------------------------------------------------------

# ML Problem Definition

Frame the task as **two binary classification problems** using a single
snapshot dataset with both labels.

Predict whether a resident will experience either of the following in the
next 30 days:

- **Fall** (from incidents)
- **Return-to-hospital** (from hospital_admissions only)

Target variables:

- label_fall_30d
- label_rth_30d

Definitions:

- label_fall_30d = 1 if a fall occurs within next 30 days else 0
- label_rth_30d = 1 if a hospital admission occurs within next 30 days else 0

Each prediction produces **two resident risk scores** (one per label).

Hospital transfers are **features only** to enrich context, not labels.

------------------------------------------------------------------------

# Dataset Location

Raw data is stored in:

data/raw/

Tables include:

adl_responses.parquet\
care_plans.parquet\
diagnoses.parquet\
document_tags.parquet\
factors.parquet\
gg_responses.parquet\
hospital_admissions.parquet\
hospital_transfers.parquet\
incidents.parquet\
injuries.parquet\
lab_reports.parquet\
medications.parquet\
needs.parquet\
physician_orders.parquet\
residents.parquet\
therapy_tracks.parquet\
vitals.parquet

Primary entity:

resident_id

Many datasets also include timestamps.

------------------------------------------------------------------------

# Project Structure

The project must follow this structure:

tricura-incident-risk
├── AGENT_INSTRUCTIONS.md
├── README.md
├── data/
│   └── raw/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_datavision_eda.ipynb
│   ├── 03_rule_based.ipynb
│   ├── 04_decision_tree.ipynb
│   ├── 05_lgb_fall.ipynb
│   └── 06_lgb_rth.ipynb
├── outputs/
│   ├── datavision_weekly_2025.parquet
│   └── datavision_weekly_2023-08_2025-01.parquet
├── reports/
│   ├── model_metrics.csv
│   ├── feature_importance.csv
│   └── plots/
├── models/
│   ├── fall_rule_model.json
│   ├── rth_rule_model.json
│   ├── fall_tree_model.pkl
│   ├── rth_tree_model.pkl
│   ├── fall_lgb_model.pkl
│   └── rth_lgb_model.pkl
├── src/
│   ├── data/
│   │   ├── build_dataset.py
│   │   ├── build_snapshots.py
│   │   └── load_raw.py
│   ├── features/
│   │   ├── event_features.py
│   │   └── labels.py
│   └── utils/
│       └── time_utils.py
└── requirements.txt

------------------------------------------------------------------------

# Development Plan

The system should be implemented in stages.

1.  Data loading
2.  Exploratory data analysis
3.  Resident timeline construction
4.  Feature engineering
5.  Dataset creation
6.  Model training (two models, shared dataset)
7.  Model evaluation
8.  Model interpretation

------------------------------------------------------------------------

# Step 1 --- Data Loading

File:

src/data/load_data.py

Responsibilities:

-   Load all parquet files
-   Store them in a dictionary
-   Return clean pandas DataFrames

Example function:

load_all_tables(data_path) -\> dict\[str, pd.DataFrame\]

Libraries:

pandas\
pyarrow

------------------------------------------------------------------------

# Step 2 --- Exploratory Data Analysis

Notebook:

notebooks/01_eda.ipynb

Tasks:

-   Load all datasets
-   Print schema
-   Show row counts
-   Identify time columns
-   Identify keys and joins
-   Identify missing values

Visualizations:

-   incident frequency
-   incident type distribution
-   incidents over time

------------------------------------------------------------------------

# Step 3 --- Build Resident Timeline

File:

src/data/build_timeline.py

Goal:

Create a **time-aware modeling dataset**.

Each row represents:

resident_id\
snapshot_date\
features\
labels

Rules:

-   Features must use **only past information**
-   Labels must represent **future events**
-   Prevent temporal leakage

------------------------------------------------------------------------

# Step 4 --- Feature Engineering

Files:

src/features/event_features.py
src/features/labels.py

Generate features from multiple tables.

Feature groups include:

Incident History

incidents_last_7_days\
incidents_last_30_days\
incidents_last_90_days\
days_since_last_incident

Medication Features

num_active_medications\
recent_medication_change\
polypharmacy_indicator

Diagnosis Features

num_diagnoses\
dementia_indicator\
chronic_condition_count

Hospitalization Features

recent_hospitalization\
days_since_last_admission\
hospitalizations_last_6_months

Functional Status

adl_score\
mobility_score\
functional_dependence_index

Vital Signs

recent_bp\
recent_hr\
bp_variability\
vital_instability_score

Therapy Indicators

physical_therapy_active\
occupational_therapy_active

Care Plan Indicators

fall_risk_care_plan\
wound_care_plan\
monitoring_flag

------------------------------------------------------------------------

# Step 5 --- Dataset Creation

Combine engineered features into a single dataset.

Output files:

outputs/datavision_weekly_2025.parquet
outputs/datavision_weekly_2023-08_2025-01.parquet

Columns:

resident_id\
reference_date\
features...\
incident_next_30_days

------------------------------------------------------------------------

# Step 6 --- Model Training

Notebooks:

notebooks/03_rule_based.ipynb
notebooks/04_decision_tree.ipynb
notebooks/05_lgb_fall.ipynb
notebooks/06_lgb_rth.ipynb

Models:

- Rule-based baseline
- Decision tree (interpretable)
- LightGBM (high-performance)

------------------------------------------------------------------------

# Step 7 --- Validation Strategy

Use **temporal validation**.

Example:

train = first 80% of timeline\
test = last 20%

Never use random splits.

------------------------------------------------------------------------

# Step 8 --- Evaluation

File:

src/models/evaluate_model.py

Metrics:

ROC-AUC\
PR-AUC

Operational metric:

Recall@Top10%

Example interpretation:

Top 10% highest risk residents capture 40% of incidents.

------------------------------------------------------------------------

# Step 9 --- Model Interpretation

Use SHAP.

Goal:

Identify drivers of incidents.

Likely drivers:

previous falls\
mobility decline\
recent hospitalization\
polypharmacy\
dementia

Save plots to:

outputs/

------------------------------------------------------------------------

# Step 10 --- Business Impact

Estimate potential financial impact.

Example:

Average fall cost = \$3,500\
Return-to-hospital cost = \$20,000

Estimate savings if monitoring **top risk residents reduces incidents**.

------------------------------------------------------------------------

# Coding Guidelines

-   Use pandas for transformations
-   Keep functions modular
-   Avoid complex pipelines
-   Write clear docstrings
-   Log key assumptions

------------------------------------------------------------------------

# Required Libraries

pandas\
numpy\
pyarrow\
scikit-learn\
lightgbm\
matplotlib\
seaborn\
shap

------------------------------------------------------------------------

# Deliverables

The final repository should contain:

-   clean modular code
-   reproducible training pipeline
-   EDA notebook
-   evaluation results
-   feature importance plots
-   clear README

------------------------------------------------------------------------

# README Requirements

README must include:

1.  Business problem
2.  Modeling approach
3.  Feature engineering strategy
4.  Model performance
5.  Key insights
6.  Potential business impact

------------------------------------------------------------------------

# Success Criteria

A strong solution demonstrates:

-   strong ML problem framing
-   meaningful feature engineering
-   correct temporal validation
-   interpretable models
-   actionable insights
