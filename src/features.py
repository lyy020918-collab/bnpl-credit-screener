"""Feature engineering for the BNPL credit screener.

Both training and the live app call these functions, so a single applicant row
goes through exactly the same transformations as the training data. Every
feature here is computed row-wise (ratios, per-row aggregates), so it works the
same on one row or on the whole table.
"""
import numpy as np
import pandas as pd

EXT_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

# Names of the engineered columns (handy for reference / documentation).
ENGINEERED = [
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
    "DAYS_EMPLOYED_PERC", "EXT_SOURCE_MEAN", "EXT_SOURCE_STD",
    "EXT_SOURCE_MIN", "EXT_SOURCE_MAX", "EXT_SOURCE_COUNT", "INCOME_PER_PERSON",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues in the raw Home Credit table."""
    df = df.copy()
    # 365243 is the sentinel Home Credit uses for "not employed" in DAYS_EMPLOYED
    # (about 1000 years). Left as-is it poisons every ratio, so treat it as missing.
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Safe on a single-row frame."""
    df = df.copy()
    inc = df["AMT_INCOME_TOTAL"].replace(0, np.nan)

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / inc
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / inc
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].replace(0, np.nan)
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]

    ext = df[EXT_COLS]
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_STD"] = ext.std(axis=1)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1)
    df["EXT_SOURCE_MAX"] = ext.max(axis=1)
    # How many of the 3 external credit scores are present — the thin-file signal.
    df["EXT_SOURCE_COUNT"] = ext.notna().sum(axis=1)

    if "CNT_FAM_MEMBERS" in df.columns:
        fam = df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    else:
        fam = 1.0
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / fam

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """clean + add_features in one call."""
    return add_features(clean(df))


def thin_file_mask(df: pd.DataFrame) -> pd.Series:
    """Thin-file = at most ONE of the three external credit scores is available.

    This is our working proxy for applicants with little traditional credit
    history — the population the screener is meant to serve.
    """
    return df[EXT_COLS].notna().sum(axis=1) <= 1
