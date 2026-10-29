"""BNPL Credit Screener — Streamlit app.

Scores a thin-file applicant's default risk with a gradient-boosting model and
explains each decision. Run locally:  streamlit run app.py
"""
import os
import streamlit as st

from src import config as C
from src import predict as P

st.set_page_config(page_title="BNPL Credit Screener", page_icon="💳", layout="centered")

st.title("💳 BNPL Credit Screener")
st.caption(
    "Academic prototype for FINS5557 — **not** a real lending tool. "
    "Do not enter real personal data."
)

# ---- Load model artifacts (graceful error if not trained yet) --------------
try:
    art = P.load_artifacts()
    meta = art["meta"]
except Exception:
    st.error(
        "Model artifacts not found. Train the model first:\n\n"
        "`python -m src.train --data data/application_train.csv`"
    )
    st.stop()

with st.expander("Model performance (held-out test set)", expanded=False):
    m = meta["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC (gradient boosting)", f"{m['hgb_auc']:.3f}")
    c2.metric("ROC-AUC (logistic baseline)", f"{m['lr_auc']:.3f}")
    c3.metric("Default rate", f"{m['default_rate']:.1%}")
    st.caption(
        f"Majority-class accuracy is {m['majority_accuracy']:.1%}, which is why "
        "accuracy is misleading under class imbalance and we report AUC instead."
    )

st.subheader("Applicant details")

# ---- Build the input form --------------------------------------------------
user_inputs = {}
with st.form("applicant"):
    cols = st.columns(2)
    for i, f in enumerate(C.EXPOSED_FEATURES):
        widget = cols[i % 2]
        col, kind, label = f["col"], f["kind"], f["label"]
        if kind == "category":
            choices = meta["categories"].get(col, [])
            default = meta["defaults"].get(col)
            idx = choices.index(default) if default in choices else 0
            user_inputs[col] = widget.selectbox(label, choices, index=idx)
        else:
            user_inputs[col] = widget.number_input(
                label, min_value=float(f["min"]), max_value=float(f["max"]),
                value=float(f["default"]), step=float(f["step"]))
    submitted = st.form_submit_button("Score applicant")

# ---- Score + explain -------------------------------------------------------
if submitted:
    # Basic input validation.
    errors = []
    if user_inputs["AMT_INCOME_TOTAL"] <= 0:
        errors.append("Annual income must be greater than 0.")
    if user_inputs["AMT_CREDIT"] <= 0:
        errors.append("Credit amount must be greater than 0.")
    if errors:
        for e in errors:
            st.warning(e)
        st.stop()

    try:
        res = P.predict(user_inputs)
    except Exception as e:  # noqa: BLE001
        st.error(f"Scoring failed: {e}")
        st.stop()

    proba, th, decision = res["proba"], res["threshold"], res["decision"]
    st.subheader("Decision")
    c1, c2 = st.columns(2)
    c1.metric("Predicted default probability", f"{proba:.1%}")
    c2.metric("Decision threshold", f"{th:.0%}")
    if decision == "APPROVE":
        st.success(f"**{decision}** — risk below the cost-sensitive threshold.")
    else:
        st.error(f"**{decision}** — risk at or above the cost-sensitive threshold.")

    st.subheader("Why this decision")
    exp = P.explain(user_inputs)
    st.caption(f"Explanation method: {exp['method']}. "
               "Bars to the right push risk up; to the left, down.")
    for label, val in exp["contributions"]:
        direction = "🔴 increases risk" if val > 0 else "🟢 lowers risk"
        st.write(f"- **{label}** — {direction}  ({val:+.3f})")

    st.info(
        "This explanation is indicative. The model is a prototype trained on the "
        "Home Credit dataset and must not be used for real credit decisions."
    )
