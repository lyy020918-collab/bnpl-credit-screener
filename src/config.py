"""Central configuration: paths and the set of applicant inputs the app exposes.

The model is trained on all ~120 columns of the Home Credit application table.
The Streamlit app, however, only asks the user for a small, intuitive subset of
features (below). Every other column is filled with a default value (median for
numeric, mode for categorical) that is learned at training time and saved in the
model metadata. This keeps the UI usable while still feeding the model the full
feature vector it expects.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# Default dataset location (not committed to the repo — see README / .gitignore).
DATA_FILE = DATA_DIR / "application_train.csv"

# Main gradient-boosting model (XGBoost when available, else HistGradientBoosting).
MODEL_GB = MODELS_DIR / "model_gb.joblib"
MODEL_LR = MODELS_DIR / "model_lr.joblib"
MODEL_RF = MODELS_DIR / "model_rf.joblib"
METADATA = MODELS_DIR / "metadata.joblib"

TARGET = "TARGET"
DROP_COLS = ["TARGET", "SK_ID_CURR"]

# Cost ratio for the cost-sensitive decision threshold.
# A missed defaulter (false negative) is assumed to cost C_FN times as much as
# wrongly rejecting a good applicant (false positive). Tune this with the team.
COST_FN = 5.0
COST_FP = 1.0

# Risk-appetite grid. The app exposes these cost ratios (C_FN : C_FP) as a
# slider so a risk officer can trade off approving good customers against
# catching defaulters. At training time we precompute the cost-optimal decision
# threshold for every ratio (both for the full population and for the thin-file
# "no external score" regime) and store it in metadata, so the app can look up a
# threshold for any slider position without needing the test data at runtime.
RISK_RATIOS = [5, 8, 10, 12, 15]
DEFAULT_RISK_RATIO = 10

# ---------------------------------------------------------------------------
# Features the app asks the user for. Each entry drives one input widget.
#   kind: "number" | "category" | "age_years" | "employ_years"
#   age_years / employ_years are shown in human units and converted back to the
#   raw Home Credit DAYS_* representation before being passed to the model.
# ---------------------------------------------------------------------------
EXPOSED_FEATURES = [
    {"col": "AMT_INCOME_TOTAL", "label": "Annual income", "kind": "number",
     "min": 0.0, "max": 1_000_000.0, "step": 1000.0, "default": 150000.0},
    {"col": "AMT_CREDIT", "label": "Credit amount requested", "kind": "number",
     "min": 0.0, "max": 2_000_000.0, "step": 1000.0, "default": 500000.0},
    {"col": "AMT_ANNUITY", "label": "Loan annuity (per period)", "kind": "number",
     "min": 0.0, "max": 200000.0, "step": 500.0, "default": 25000.0},
    {"col": "DAYS_BIRTH", "label": "Age (years)", "kind": "age_years",
     "min": 18, "max": 100, "step": 1, "default": 40},
    {"col": "DAYS_EMPLOYED", "label": "Years in current employment", "kind": "employ_years",
     "min": 0, "max": 50, "step": 1, "default": 5},
    {"col": "EXT_SOURCE_2", "label": "External credit score 2 (0-1)", "kind": "number",
     "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.5},
    {"col": "EXT_SOURCE_3", "label": "External credit score 3 (0-1)", "kind": "number",
     "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.5},
    {"col": "CNT_CHILDREN", "label": "Number of children", "kind": "number",
     "min": 0, "max": 20, "step": 1, "default": 0},
    {"col": "CODE_GENDER", "label": "Gender", "kind": "category"},
    {"col": "NAME_EDUCATION_TYPE", "label": "Education", "kind": "category"},
    {"col": "NAME_FAMILY_STATUS", "label": "Family status", "kind": "category"},
    {"col": "NAME_INCOME_TYPE", "label": "Income type", "kind": "category"},
    {"col": "NAME_CONTRACT_TYPE", "label": "Contract type", "kind": "category"},
]

# Human-unit <-> raw conversion for the two DAYS_* fields.
def years_to_days_birth(age_years: float) -> float:
    return -abs(age_years) * 365.0

def years_to_days_employed(years: float) -> float:
    return -abs(years) * 365.0
