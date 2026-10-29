"""Train the BNPL credit-screening models and save artifacts.

Usage:
    python -m src.train                       # uses data/application_train.csv
    python -m src.train --data /path/file.csv  # custom path

Produces, in models/:
    model_hgb.joblib   HistGradientBoosting pipeline (main model)
    model_lr.joblib    LogisticRegression pipeline (interpretable baseline)
    metadata.joblib    feature order, defaults, categories, threshold, metrics

A global SHAP summary plot is also written to models/shap_summary.png when the
`shap` package is installed (it is listed in requirements.txt and available on
Streamlit Cloud / Colab). Training here does not depend on it.
"""
import argparse, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, recall_score)

from . import config as C


def build_models(num_cols, cat_cols, cat_mask):
    num_lr = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc", StandardScaler())])
    cat_lr = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                       ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))])
    pre_lr = ColumnTransformer([("num", num_lr, num_cols), ("cat", cat_lr, cat_cols)])
    lr = Pipeline([("pre", pre_lr),
                   ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

    pre_gb = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-1), cat_cols)],
        remainder="passthrough")
    gb = Pipeline([("pre", pre_gb),
                   ("clf", HistGradientBoostingClassifier(
                       max_iter=300, learning_rate=0.05, l2_regularization=1.0,
                       categorical_features=cat_mask, class_weight="balanced",
                       random_state=42))])
    return lr, gb


def cost_sensitive_threshold(y_true, proba):
    ths = np.linspace(0.01, 0.99, 99)
    best_th, best_cost = 0.5, np.inf
    for th in ths:
        pred = (proba >= th).astype(int)
        fn = ((pred == 0) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        cost = C.COST_FN * fn + C.COST_FP * fp
        if cost < best_cost:
            best_cost, best_th = cost, th
    return float(best_th)


def main(data_path):
    t0 = time.time()
    df = pd.read_csv(data_path)
    y = df[C.TARGET].values
    X = df.drop(columns=[c for c in C.DROP_COLS if c in df.columns])

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    feature_order = X.columns.tolist()
    # After the GB ColumnTransformer, categorical columns come first, then numeric.
    cat_mask = [True] * len(cat_cols) + [False] * len(num_cols)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    lr, gb = build_models(num_cols, cat_cols, cat_mask)
    print("training logistic regression ...")
    lr.fit(Xtr, ytr)
    print("training hist gradient boosting ...")
    gb.fit(Xtr, ytr)

    p_lr = lr.predict_proba(Xte)[:, 1]
    p_gb = gb.predict_proba(Xte)[:, 1]
    metrics = {
        "default_rate": float(yte.mean()),
        "majority_accuracy": float(accuracy_score(yte, np.zeros_like(yte))),
        "lr_auc": float(roc_auc_score(yte, p_lr)),
        "lr_pr_auc": float(average_precision_score(yte, p_lr)),
        "hgb_auc": float(roc_auc_score(yte, p_gb)),
        "hgb_pr_auc": float(average_precision_score(yte, p_gb)),
    }
    threshold = cost_sensitive_threshold(yte, p_gb)
    pred = (p_gb >= threshold).astype(int)
    metrics["threshold"] = threshold
    metrics["recall_at_threshold"] = float(recall_score(yte, pred))
    metrics["approval_rate"] = float((pred == 0).mean())

    # Defaults for columns the app does not expose.
    defaults = {}
    for c in num_cols:
        defaults[c] = float(X[c].median())
    for c in cat_cols:
        m = X[c].mode(dropna=True)
        defaults[c] = (m.iloc[0] if len(m) else "")
    categories = {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols}

    C.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(gb, C.MODEL_HGB)
    joblib.dump(lr, C.MODEL_LR)
    joblib.dump({
        "feature_order": feature_order,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "defaults": defaults,
        "categories": categories,
        "threshold": threshold,
        "metrics": metrics,
    }, C.METADATA)

    print("\n--- metrics ---")
    print(json.dumps(metrics, indent=2))
    print(f"\nsaved models to {C.MODELS_DIR}")

    # Optional global SHAP summary (only if shap is installed).
    try:
        _save_shap_summary(gb, Xtr, Xte, cat_cols)
    except Exception as e:  # noqa: BLE001
        print(f"[shap] skipped global summary: {e}")

    print(f"elapsed {time.time() - t0:.0f}s")


def _save_shap_summary(gb_pipeline, Xtr, Xte, cat_cols, n=1000):
    import shap, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pre = gb_pipeline.named_steps["pre"]
    model = gb_pipeline.named_steps["clf"]
    Xte_t = pre.transform(Xte.iloc[:n])
    # Column order after transform: categorical first, then numeric.
    cols = cat_cols + [c for c in Xte.columns if c not in cat_cols]
    explainer = shap.Explainer(model, pre.transform(Xtr.iloc[:n]))
    sv = explainer(Xte_t)
    sv.feature_names = cols
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(C.MODELS_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[shap] wrote {C.MODELS_DIR / 'shap_summary.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DATA_FILE))
    main(ap.parse_args().data)
