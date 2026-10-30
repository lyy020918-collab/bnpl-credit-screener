"""BNPL Credit Screener — Streamlit app.

Scores a thin-file applicant's default risk with an XGBoost gradient-boosting
model and explains each decision. Run locally:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

try:
    import altair as alt
    _HAS_ALT = True
except Exception:  # noqa: BLE001
    _HAS_ALT = False

from src import config as C
from src import predict as P

st.set_page_config(page_title="BNPL Credit Screener", page_icon="💳",
                   layout="centered")


# --------------------------------------------------------------------------
# Performance optimisation: load the model + metadata ONCE and cache them.
# The XGBoost model and preprocessing pipeline together are a few MB; loading
# them on every rerun (every widget change) would add noticeable latency.
# @st.cache_resource keeps a single copy in memory, shared across reruns and
# across users, so each interaction only runs inference, not a reload.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def get_artifacts():
    return P.load_artifacts()


st.title("💳 BNPL Credit Screener")
st.caption(
    "Academic prototype for FINS5557 — **not** a real lending tool. "
    "Do not enter real personal data."
)

try:
    art = get_artifacts()
    meta = art["meta"]
    m = meta["metrics"]
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Train the model first:\n\n"
        "`python -m src.train --data data/application_train.csv`"
    )
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error("Could not load the model — details below:")
    st.exception(e)
    st.stop()

# ---- Sidebar: instructions + model summary --------------------------------
with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Enter the applicant's details.\n"
        "2. Click **Score applicant**.\n"
        "3. Read the decision and the reasons behind it."
    )
    st.divider()
    st.header("Model")
    gb = meta.get("gb_name", "XGBoost")
    st.metric(f"ROC-AUC ({gb})", f"{m['models'][gb]['auc']:.3f}")
    st.metric("ROC-AUC (logistic baseline)", f"{m['models']['LogisticRegression']['auc']:.3f}")
    st.metric("Thin-file subset ROC-AUC", f"{m['thin_file_auc']:.3f}")
    st.caption(
        f"Default rate {m['default_rate']:.1%}; majority-class accuracy "
        f"{m['majority_accuracy']:.1%}. Accuracy is misleading under this class "
        "imbalance, so we report AUC."
    )

st.write(
    "Enter an applicant below. The screener estimates default risk from "
    "alternative and behavioural data, then explains the decision — useful for "
    "**thin-file** applicants with little traditional credit history."
)

# ---- External-scores toggle (the thin-file switch) ------------------------
# External bureau scores are the strongest predictors, but thin-file applicants
# often have none. Unchecking this scores the applicant on alternative data only.
EXT_INPUTS = ("EXT_SOURCE_2", "EXT_SOURCE_3")
has_ext = st.checkbox(
    "Applicant has external credit-bureau scores",
    value=True,
    help="External bureau scores (0–1, higher = lower risk) are the strongest "
         "predictors. Thin-file applicants often have none — uncheck to score "
         "the applicant on alternative and behavioural data only.",
)
if not has_ext:
    st.info("Scoring as a **thin-file** applicant — no external bureau scores available.")

# ---- Input form -----------------------------------------------------------
user_inputs = {}
with st.form("applicant"):
    cols = st.columns(2)
    slot = 0
    for f in C.EXPOSED_FEATURES:
        col, kind, label = f["col"], f["kind"], f["label"]
        if col in EXT_INPUTS and not has_ext:
            continue  # hidden; set to missing after the form
        widget = cols[slot % 2]
        slot += 1
        if kind == "category":
            choices = meta["categories"].get(col, [])
            default = meta["defaults"].get(col)
            idx = choices.index(default) if default in choices else 0
            user_inputs[col] = widget.selectbox(label, choices, index=idx)
        else:
            help_txt = ("External bureau score, 0–1 (higher = lower risk)."
                        if col in EXT_INPUTS else None)
            user_inputs[col] = widget.number_input(
                label, min_value=float(f["min"]), max_value=float(f["max"]),
                value=float(f["default"]), step=float(f["step"]), help=help_txt)
    submitted = st.form_submit_button("Score applicant", type="primary")

# No external scores → represent a true thin-file applicant: all three missing.
if not has_ext:
    for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        user_inputs[c] = float("nan")

if not submitted:
    st.stop()

# ---- Validate -------------------------------------------------------------
errors = []
if user_inputs["AMT_INCOME_TOTAL"] <= 0:
    errors.append("Annual income must be greater than 0.")
if user_inputs["AMT_CREDIT"] <= 0:
    errors.append("Credit amount must be greater than 0.")
if errors:
    for e in errors:
        st.warning(e)
    st.stop()

# ---- Score (loading state) ------------------------------------------------
try:
    with st.spinner("Scoring applicant…"):
        res = P.predict(user_inputs)
except Exception as e:  # noqa: BLE001
    st.error(f"Scoring failed: {e}")
    st.stop()

proba, th, decision = res["proba"], res["threshold"], res["decision"]

st.subheader("Decision")
c1, c2 = st.columns(2)
c1.metric("Predicted default probability", f"{proba:.1%}")
c2.metric("Decision threshold", f"{th:.0%}")
st.progress(min(proba, 1.0))
if decision == "APPROVE":
    st.success(f"✅ **{decision}** — risk is below the cost-sensitive threshold.")
else:
    st.error(f"⛔ **{decision}** — risk is at or above the cost-sensitive threshold.")

# ---- Explain (loading state) ----------------------------------------------
st.subheader("Why this decision")
try:
    with st.spinner("Computing explanation…"):
        exp = P.explain(user_inputs)
except Exception as e:  # noqa: BLE001
    st.warning(f"Explanation unavailable: {e}")
    st.stop()

st.caption(
    f"Method: {exp['method']}. Red bars push risk up; green bars push it down."
)
df = pd.DataFrame(exp["contributions"], columns=["feature", "contribution"])
df["direction"] = df["contribution"].apply(
    lambda v: "Increases risk" if v > 0 else "Lowers risk")

if _HAS_ALT and len(df):
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="Impact on predicted risk"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Increases risk", "Lowers risk"],
                                range=["#d62728", "#2ca02c"]),
                legend=None),
            tooltip=["feature", alt.Tooltip("contribution:Q", format="+.3f")],
        )
        .properties(height=28 * len(df))
    )
    st.altair_chart(chart, use_container_width=True)
else:
    for label, val in exp["contributions"]:
        tag = "🔴 increases risk" if val > 0 else "🟢 lowers risk"
        st.write(f"- **{label}** — {tag} ({val:+.3f})")

st.info(
    "This explanation is indicative. The model is a prototype trained on the "
    "Home Credit dataset and must not be used for real credit decisions."
)
