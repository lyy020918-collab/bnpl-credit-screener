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


def build_models(num_cols, cat_cols, cat_mask):
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

    # HistGradientBoosting: ordinal cats (native NaN handling for numerics).
    pre_gb = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-1), cat_cols)],
        remainder="passthrough")
    gb = Pipeline([("pre", pre_gb),
                   ("clf", HistGradientBoostingClassifier(
                       max_iter=400, learning_rate=0.05, l2_regularization=1.0,
                       categorical_features=cat_mask, class_weight="balanced",
                       random_state=42))])
    return {"LogisticRegression": lr, "RandomForest": rf, "HistGradientBoosting": gb}


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

    models = build_models(num_cols, cat_cols, cat_mask)
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

    # Best of the three real models = app's main model.
    real = ["LogisticRegression", "RandomForest", "HistGradientBoosting"]
    best_name = max(real, key=lambda n: metrics["models"][n]["auc"])
    best_p = proba[best_name]
    threshold = cost_sensitive_threshold(yte, best_p)
    pred = (best_p >= threshold).astype(int)
    metrics["best_model"] = best_name
    metrics["threshold"] = threshold
    metrics["recall_at_threshold"] = float(recall_score(yte, pred))
    metrics["approval_rate"] = float((pred == 0).mean())

    # Thin-file subset performance (best model).
    thin = F.thin_file_mask(Xte).values
    metrics["thin_file_share"] = float(thin.mean())
    metrics["thin_file_auc"] = float(roc_auc_score(yte[thin], best_p[thin]))
    metrics["non_thin_file_auc"] = float(roc_auc_score(yte[~thin], best_p[~thin]))
    metrics["thin_file_default_rate"] = float(yte[thin].mean())

    # ---- Figures ----
    _fig_roc(yte, proba, real)
    _fig_pr(yte, proba, real)
    _fig_confusion(yte, pred, best_name, threshold)
    _fig_calibration(yte, best_p, best_name)
    _fig_perm_importance(fitted[best_name], Xte, yte, best_name)

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

    joblib.dump(fitted["HistGradientBoosting"], C.MODEL_HGB)
    joblib.dump(fitted["LogisticRegression"], C.MODEL_LR)
    joblib.dump(fitted["RandomForest"], C.MODELS_DIR / "model_rf.joblib")
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
        "best_model": best_name,
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DATA_FILE))
    main(ap.parse_args().data)
