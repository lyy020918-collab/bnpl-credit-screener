"""Load trained artifacts, score a single applicant, and explain the decision.

Explanation strategy:
  * If `shap` is installed (deployment / Colab), produce true SHAP values for the
    single applicant over the features the app exposed.
  * Otherwise fall back to a model-agnostic local importance: for each exposed
    feature, measure how much the predicted probability moves when that feature
    is reset to its training default. This needs no extra dependency and keeps
    the app fully functional everywhere.
"""
from __future__ import annotations
import joblib, numpy as np, pandas as pd

from . import config as C

_ARTIFACTS = None


def load_artifacts():
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = {
            "hgb": joblib.load(C.MODEL_HGB),
            "lr": joblib.load(C.MODEL_LR),
            "meta": joblib.load(C.METADATA),
        }
    return _ARTIFACTS


def build_feature_row(user_inputs: dict, meta: dict) -> pd.DataFrame:
    """Turn the handful of app inputs into a full one-row feature frame."""
    row = dict(meta["defaults"])  # start from training defaults
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
    return pd.DataFrame([[row[c] for c in meta["feature_order"]]],
                        columns=meta["feature_order"])


def predict(user_inputs: dict):
    art = load_artifacts()
    meta = art["meta"]
    X1 = build_feature_row(user_inputs, meta)
    proba = float(art["hgb"].predict_proba(X1)[:, 1][0])
    threshold = meta["threshold"]
    decision = "DECLINE / refer" if proba >= threshold else "APPROVE"
    return {"proba": proba, "threshold": threshold, "decision": decision, "X1": X1}


def explain(user_inputs: dict, top_k: int = 8):
    """Return a list of (feature_label, signed_contribution) for the applicant."""
    art = load_artifacts()
    meta = art["meta"]
    X1 = build_feature_row(user_inputs, meta)
    labels = {f["col"]: f["label"] for f in C.EXPOSED_FEATURES}
    exposed_cols = [f["col"] for f in C.EXPOSED_FEATURES]

    try:
        return _shap_explain(art, meta, X1, exposed_cols, labels, top_k)
    except Exception:
        return _ablation_explain(art, meta, X1, exposed_cols, labels, top_k)


def _shap_explain(art, meta, X1, exposed_cols, labels, top_k):
    import shap
    gb = art["hgb"]
    pre, model = gb.named_steps["pre"], gb.named_steps["clf"]
    cols = meta["cat_cols"] + [c for c in meta["feature_order"]
                               if c not in meta["cat_cols"]]
    bg = pre.transform(pd.DataFrame(
        [[meta["defaults"][c] for c in meta["feature_order"]]],
        columns=meta["feature_order"]))
    explainer = shap.Explainer(model, bg)
    sv = explainer(pre.transform(X1))
    vals = np.array(sv.values).reshape(-1)
    contrib = {cols[i]: float(vals[i]) for i in range(len(cols))}
    pairs = [(labels[c], contrib.get(c, 0.0)) for c in exposed_cols]
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return {"method": "SHAP", "contributions": pairs[:top_k]}


def _ablation_explain(art, meta, X1, exposed_cols, labels, top_k):
    gb = art["hgb"]
    base = float(gb.predict_proba(X1)[:, 1][0])
    pairs = []
    for col in exposed_cols:
        X_mod = X1.copy()
        X_mod[col] = meta["defaults"][col]
        p_mod = float(gb.predict_proba(X_mod)[:, 1][0])
        # Positive contribution = this feature pushes risk UP vs the default.
        pairs.append((labels[col], base - p_mod))
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return {"method": "local ablation (install shap for full SHAP)",
            "contributions": pairs[:top_k]}
