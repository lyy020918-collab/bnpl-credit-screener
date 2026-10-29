"""Load trained artifacts, score a single applicant, and explain the decision.

Explanation strategy:
  * If `shap` is installed (deployment / Colab), produce true SHAP values for the
    applicant and show the features that moved the decision most.
  * Otherwise fall back to a model-agnostic local importance: reset each feature
    to its training median and measure how far the predicted probability moves.
    Needs no extra dependency, so the app stays fully functional everywhere.
"""
from __future__ import annotations
import joblib, numpy as np, pandas as pd

from . import config as C
from . import features as F

_ARTIFACTS = None

# Friendly names for the features most likely to appear in an explanation.
LABELS = {
    "EXT_SOURCE_MEAN": "External credit score (average)",
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "EXT_SOURCE_MIN": "External credit score (lowest)",
    "EXT_SOURCE_MAX": "External credit score (highest)",
    "EXT_SOURCE_COUNT": "Number of external scores available",
    "CREDIT_INCOME_RATIO": "Credit-to-income ratio",
    "ANNUITY_INCOME_RATIO": "Annuity-to-income ratio",
    "CREDIT_TERM": "Loan term (annuity / credit)",
    "DAYS_EMPLOYED_PERC": "Share of life in current job",
    "INCOME_PER_PERSON": "Income per household member",
    "AMT_CREDIT": "Credit amount requested",
    "AMT_INCOME_TOTAL": "Annual income",
    "AMT_ANNUITY": "Loan annuity",
    "DAYS_BIRTH": "Age",
    "DAYS_EMPLOYED": "Employment length",
    "NAME_EDUCATION_TYPE": "Education",
    "NAME_FAMILY_STATUS": "Family status",
    "NAME_INCOME_TYPE": "Income type",
    "CODE_GENDER": "Gender",
}


def label(col: str) -> str:
    return LABELS.get(col, col.replace("_", " ").title())


def load_artifacts():
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = {
            "gb": joblib.load(C.MODEL_GB),
            "lr": joblib.load(C.MODEL_LR),
            "meta": joblib.load(C.METADATA),
        }
    return _ARTIFACTS


def build_feature_row(user_inputs: dict, meta: dict) -> pd.DataFrame:
    """Build a full one-row feature frame (raw defaults + inputs -> engineered)."""
    row = {c: meta["defaults"][c] for c in meta["raw_cols"]}
    for f in C.EXPOSED_FEATURES:
        col, kind = f["col"], f["kind"]
        if col not in user_inputs:
            continue
        val = user_inputs[col]
        if kind == "age_years":
            row[col] = C.years_to_days_birth(val)
        elif kind == "employ_years":
            row[col] = C.years_to_days_employed(val)
        else:
            row[col] = val
    raw_df = pd.DataFrame([[row[c] for c in meta["raw_cols"]]], columns=meta["raw_cols"])
    prepared = F.prepare(raw_df)
    return prepared.reindex(columns=meta["feature_order"])


def predict(user_inputs: dict):
    art = load_artifacts()
    meta = art["meta"]
    X1 = build_feature_row(user_inputs, meta)
    proba = float(art["gb"].predict_proba(X1)[:, 1][0])
    threshold = meta["threshold"]
    decision = "DECLINE / refer" if proba >= threshold else "APPROVE"
    return {"proba": proba, "threshold": threshold, "decision": decision, "X1": X1}


def explain(user_inputs: dict, top_k: int = 8):
    """Return {method, contributions:[(label, signed_value), ...]} for the applicant."""
    art = load_artifacts()
    meta = art["meta"]
    X1 = build_feature_row(user_inputs, meta)
    try:
        return _shap_explain(art, meta, X1, top_k)
    except Exception:
        return _ablation_explain(art, meta, X1, top_k)


def _shap_explain(art, meta, X1, top_k):
    import shap
    gb = art["gb"]
    pre, model = gb.named_steps["pre"], gb.named_steps["clf"]
    cols = meta["cat_cols"] + [c for c in meta["feature_order"]
                               if c not in meta["cat_cols"]]
    bg = pre.transform(pd.DataFrame(
        [[meta["defaults"].get(c, meta["feature_medians"].get(c, 0))
          for c in meta["raw_cols"]]], columns=meta["raw_cols"]).pipe(
        lambda d: F.prepare(d).reindex(columns=meta["feature_order"])))
    explainer = shap.Explainer(model, bg)
    sv = explainer(pre.transform(X1))
    vals = np.array(sv.values).reshape(-1)
    pairs = [(label(cols[i]), float(vals[i])) for i in range(len(cols))]
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return {"method": "SHAP", "contributions": pairs[:top_k]}


def _ablation_explain(art, meta, X1, top_k):
    gb = art["gb"]
    base = float(gb.predict_proba(X1)[:, 1][0])
    medians = meta["feature_medians"]
    pairs = []
    for col in meta["num_cols"]:
        if col not in medians:
            continue
        X_mod = X1.copy()
        X_mod[col] = medians[col]
        p_mod = float(gb.predict_proba(X_mod)[:, 1][0])
        delta = base - p_mod          # + = pushes risk up vs a typical applicant
        if abs(delta) > 1e-6:
            pairs.append((label(col), delta))
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return {"method": "local ablation (install shap for full SHAP)",
            "contributions": pairs[:top_k]}
