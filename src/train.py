"""Train and evaluate the BNPL credit-screening models, and save artifacts.

Usage:
    python -m src.train                        # uses data/application_train.csv
    python -m src.train --data /path/file.csv

Pipeline: clean -> engineer features -> train three models
(Logistic Regression, Random Forest, HistGradientBoosting) -> evaluate against
a majority-class baseline -> report performance on the thin-file subset
-> save models, metadata and evaluation figures.

Figures are written to figures/. Artifacts to models/.
"""
import argparse, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             recall_score, roc_curve, precision_recall_curve,
                             confusion_matrix)

from . import config as C
from . import features as F

FIG_DIR = C.ROOT / "figures"


def _make_gb(cat_cols, num_cols, cat_mask, scale_pos_weight):
    """Main gradient-boosting model: XGBoost if installed, else HistGradientBoosting.

    Both are gradient boosting (the method taught in the course). XGBoost is the
    intended model; the HistGradientBoosting fallback lets the pipeline run in
    environments where xgboost is not installed, so the code is always testable.
    Returns (name, pipeline).
    """
    # Ordinal-encode categoricals for both; both handle missing numerics natively.
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=np.nan), cat_cols)],
        remainder="passthrough")
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight, eval_metric="auc",
            tree_method="hist", random_state=42, n_jobs=-1)
        return "XGBoost", Pipeline([("pre", pre), ("clf", clf)])
    except Exception:
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, l2_regularization=1.0,
            categorical_features=cat_mask, class_weight="balanced", random_state=42)
        return "HistGradientBoosting", Pipeline([("pre", pre), ("clf", clf)])


def build_models(num_cols, cat_cols, cat_mask, scale_pos_weight):
    # Logistic Regression: impute + scale numerics, one-hot categoricals.
    num_lr = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc", StandardScaler())])
    cat_lr = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                       ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))])
    pre_lr = ColumnTransformer([("num", num_lr, num_cols), ("cat", cat_lr, cat_cols)])
    lr = Pipeline([("pre", pre_lr),
                   ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

    # Random Forest: impute numerics, ordinal-encode categoricals.
    num_rf = SimpleImputer(strategy="median")
    cat_rf = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                       ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                              unknown_value=-1))])
    pre_rf = ColumnTransformer([("num", num_rf, num_cols), ("cat", cat_rf, cat_cols)])
    rf = Pipeline([("pre", pre_rf),
                   ("clf", RandomForestClassifier(
                       n_estimators=150, max_depth=14, min_samples_leaf=40,
                       n_jobs=-1, class_weight="balanced", random_state=42))])

    # Main gradient-boosting model (XGBoost if available, else HistGradientBoosting).
    gb_name, gb = _make_gb(cat_cols, num_cols, cat_mask, scale_pos_weight)
    return {"LogisticRegression": lr, "RandomForest": rf, gb_name: gb}, gb_name


def cost_sensitive_threshold(y_true, proba, c_fn=None, c_fp=None):
    """Threshold that minimises expected cost. c_fn:c_fp is the cost of missing a
    defaulter vs. wrongly rejecting a good applicant; defaults to the config ratio."""
    c_fn = C.COST_FN if c_fn is None else c_fn
    c_fp = C.COST_FP if c_fp is None else c_fp
    ths = np.linspace(0.01, 0.99, 99)
    best_th, best_cost = 0.5, np.inf
    for th in ths:
        pred = (proba >= th).astype(int)
        fn = ((pred == 0) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        cost = c_fn * fn + c_fp * fp
        if cost < best_cost:
            best_cost, best_th = cost, th
    return float(best_th)


def main(data_path):
    t0 = time.time()
    FIG_DIR.mkdir(exist_ok=True)
    C.MODELS_DIR.mkdir(exist_ok=True)

    raw = pd.read_csv(data_path)
    raw = F.clean(raw)
    y = raw[C.TARGET].values
    raw_X = raw.drop(columns=[c for c in C.DROP_COLS if c in raw.columns])
    raw_cols = raw_X.columns.tolist()

    X = F.add_features(raw_X)                    # raw + engineered features
    feature_order = X.columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    cat_mask = [True] * len(cat_cols) + [False] * len(num_cols)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=42)

    scale_pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    models, gb_name = build_models(num_cols, cat_cols, cat_mask, scale_pos_weight)
    proba, fitted = {}, {}
    for name, pipe in models.items():
        print(f"training {name} ...")
        pipe.fit(Xtr, ytr)
        fitted[name] = pipe
        proba[name] = pipe.predict_proba(Xte)[:, 1]

    # Baseline.
    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    proba["Baseline"] = dummy.predict_proba(Xte)[:, 1]

    metrics = {"default_rate": float(yte.mean()),
               "majority_accuracy": float(accuracy_score(yte, np.zeros_like(yte))),
               "models": {}}
    for name, p in proba.items():
        metrics["models"][name] = {
            "auc": float(roc_auc_score(yte, p)),
            "pr_auc": float(average_precision_score(yte, p)),
        }

    real = ["LogisticRegression", "RandomForest", gb_name]
    metrics["best_model"] = max(real, key=lambda n: metrics["models"][n]["auc"])

    # The app always uses the gradient-boosting model, so tune the threshold and
    # report the thin-file performance on that model.
    app_p = proba[gb_name]

    thin = F.thin_file_mask(Xte).values
    metrics["thin_file_share"] = float(thin.mean())
    metrics["thin_file_auc"] = float(roc_auc_score(yte[thin], app_p[thin]))
    metrics["non_thin_file_auc"] = float(roc_auc_score(yte[~thin], app_p[~thin]))
    metrics["thin_file_default_rate"] = float(yte[thin].mean())

    # Thin-file probabilities: simulate the app's "no external score" mode on the
    # whole test set (force all external scores missing, recompute features). The
    # threshold tuned on applicants who mostly HAVE external scores is too lenient
    # when none are present, so thin-file decisions get their own threshold.
    raw_te = Xte.drop(columns=[c for c in F.ENGINEERED if c in Xte.columns])
    for c in F.EXT_COLS:
        raw_te[c] = np.nan
    Xte_tf = F.add_features(raw_te).reindex(columns=feature_order)
    p_tf = fitted[gb_name].predict_proba(Xte_tf)[:, 1]

    # Risk-appetite grid: for every cost ratio the app's slider exposes, store the
    # cost-optimal threshold for the full population and for the thin-file regime,
    # plus the approval rate / recall each one yields (handy for the report). The
    # app looks these up at runtime instead of re-tuning on data it does not have.
    def _summary(y_true, p, th):
        pr = (p >= th).astype(int)
        return {"threshold": round(th, 4),
                "approval_rate": float((pr == 0).mean()),
                "recall": float(recall_score(y_true, pr))}

    grid = {"global": {}, "thin_file": {}}
    for r in C.RISK_RATIOS:
        grid["global"][str(r)] = _summary(yte, app_p, cost_sensitive_threshold(yte, app_p, r))
        grid["thin_file"][str(r)] = _summary(yte, p_tf, cost_sensitive_threshold(yte, p_tf, r))
    metrics["threshold_grid"] = grid
    metrics["risk_ratios"] = C.RISK_RATIOS
    metrics["default_risk_ratio"] = C.DEFAULT_RISK_RATIO

    # Defaults the app starts at (the slider's default position).
    dr = str(C.DEFAULT_RISK_RATIO)
    threshold = grid["global"][dr]["threshold"]
    thin_threshold = grid["thin_file"][dr]["threshold"]
    pred = (app_p >= threshold).astype(int)
    metrics["threshold"] = threshold
    metrics["recall_at_threshold"] = grid["global"][dr]["recall"]
    metrics["approval_rate"] = grid["global"][dr]["approval_rate"]
    metrics["thin_file_threshold"] = thin_threshold
    metrics["thin_file_recall_at_threshold"] = grid["thin_file"][dr]["recall"]
    metrics["thin_file_approval_rate"] = grid["thin_file"][dr]["approval_rate"]

    # ---- Figures ----
    _fig_roc(yte, proba, real)
    _fig_pr(yte, proba, real)
    _fig_confusion(yte, pred, gb_name, threshold)
    _fig_calibration(yte, app_p, gb_name)
    _fig_perm_importance(fitted[gb_name], Xte, yte, gb_name)
    try:
        _fig_shap(fitted[gb_name], Xtr, Xte, cat_cols, gb_name)
    except Exception as e:  # noqa: BLE001
        print(f"[shap] global summary skipped ({e}); install shap to generate it")

    # ---- Defaults for app inputs (on the cleaned RAW columns) ----
    cat_set = set(cat_cols)
    defaults = {}
    for col in raw_cols:
        if col in cat_set:
            m = raw_X[col].mode(dropna=True)
            defaults[col] = (str(m.iloc[0]) if len(m) else "")
        else:
            defaults[col] = float(raw_X[col].median())
    categories = {col: sorted(map(str, raw_X[col].dropna().unique().tolist()))
                  for col in raw_cols if col in cat_set}
    # Medians for every numeric model feature (incl. engineered) — used as the
    # neutral baseline when explaining a single decision without shap.
    feature_medians = {c: float(X[c].median()) for c in num_cols}

    # Save the gradient-boosting model under a generic path so predict.py loads
    # it regardless of implementation (XGBoost or the HistGradientBoosting fallback).
    joblib.dump(fitted[gb_name], C.MODEL_GB)
    joblib.dump(fitted["LogisticRegression"], C.MODEL_LR)
    joblib.dump(fitted["RandomForest"], C.MODEL_RF)
    joblib.dump({
        "feature_order": feature_order,
        "raw_cols": raw_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "engineered": F.ENGINEERED,
        "defaults": defaults,
        "categories": categories,
        "feature_medians": feature_medians,
        "threshold": threshold,
        "thin_file_threshold": thin_threshold,
        "threshold_grid": metrics["threshold_grid"],
        "risk_ratios": C.RISK_RATIOS,
        "default_risk_ratio": C.DEFAULT_RISK_RATIO,
        "gb_name": gb_name,
        "best_model": metrics["best_model"],
        "metrics": metrics,
    }, C.METADATA)

    print("\n--- metrics ---")
    print(json.dumps(metrics, indent=2))
    print(f"\nmodels -> {C.MODELS_DIR}\nfigures -> {FIG_DIR}")
    print(f"elapsed {time.time() - t0:.0f}s")


# --------------------------- figure helpers ---------------------------------
def _fig_roc(y, proba, names):
    plt.figure(figsize=(6, 5))
    for n in names:
        fpr, tpr, _ = roc_curve(y, proba[n])
        plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y, proba[n]):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves"); plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR / "roc_curves.png", dpi=130); plt.close()


def _fig_pr(y, proba, names):
    plt.figure(figsize=(6, 5))
    for n in names:
        prec, rec, _ = precision_recall_curve(y, proba[n])
        plt.plot(rec, prec, label=f"{n} (AP={average_precision_score(y, proba[n]):.3f})")
    plt.axhline(y.mean(), color="k", ls="--", lw=1, label=f"Baseline ({y.mean():.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall curves"); plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR / "pr_curves.png", dpi=130); plt.close()


def _fig_confusion(y, pred, name, th):
    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["Approve", "Decline"]); plt.yticks([0, 1], ["No default", "Default"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion matrix — {name}\n(threshold={th:.2f})")
    plt.tight_layout(); plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=130); plt.close()


def _fig_calibration(y, p, name):
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac, "o-", label=name)
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed default rate")
    plt.title("Calibration curve"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR / "calibration.png", dpi=130); plt.close()


def _fig_perm_importance(pipe, Xte, yte, name, n=5000, top=15):
    idx = np.random.RandomState(42).choice(len(Xte), min(n, len(Xte)), replace=False)
    r = permutation_importance(pipe, Xte.iloc[idx], yte[idx], n_repeats=5,
                               random_state=42, scoring="roc_auc", n_jobs=-1)
    order = np.argsort(r.importances_mean)[-top:]
    cols = np.array(Xte.columns)[order]
    vals = r.importances_mean[order]
    plt.figure(figsize=(7, 6))
    plt.barh(range(len(cols)), vals, color="#4C78A8")
    plt.yticks(range(len(cols)), cols, fontsize=8)
    plt.xlabel("Drop in ROC-AUC when shuffled")
    plt.title(f"Permutation feature importance — {name}")
    plt.tight_layout(); plt.savefig(FIG_DIR / "feature_importance.png", dpi=130); plt.close()


def _fig_shap(pipe, Xtr, Xte, cat_cols, name, n=2000, top=15):
    """Global SHAP summary (beeswarm). Only runs when shap is installed."""
    import shap
    pre, model = pipe.named_steps["pre"], pipe.named_steps["clf"]
    cols = cat_cols + [c for c in Xte.columns if c not in cat_cols]
    Xt = pre.transform(Xte.iloc[:n])
    # TreeExplainer is fast and robust for XGBoost / HistGradientBoosting;
    # fall back to the generic Explainer if it is unavailable.
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xt)
    except Exception:  # noqa: BLE001
        sv = shap.Explainer(model, pre.transform(Xtr.iloc[:n]))(Xt).values
    if isinstance(sv, list):          # some versions return [class0, class1]
        sv = sv[1]
    plt.figure()
    shap.summary_plot(sv, Xt, feature_names=cols, max_display=top, show=False)
    plt.title(f"SHAP summary — {name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[shap] wrote {FIG_DIR / 'shap_summary.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DATA_FILE))
    main(ap.parse_args().data)
