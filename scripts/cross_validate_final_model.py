"""
Cross-validates the SHIPPED Ridge+text pipeline (the one saved to
models/mercari_pipeline_final.joblib), not just the tree-model candidates.

Why this exists: the README's model-comparison table reports 5-fold CV RMSLE
for XGBoost and LightGBM, but the final chosen Ridge+text model was only
evaluated on a single train/val split (0.5214). This closes that gap so the
headline number has the same statistical footing as the numbers it's being
compared against.

Usage (from repo root, after running src/data_prep.py so data/processed/
exists):
    python scripts/cross_validate_final_model.py

Requires data/processed/train.csv, which requires the raw Kaggle dataset --
neither ships with the repo (see Getting Started in the README), so this
must be run locally with the data in place. It is not run in CI.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pipeline_components import LeakSafeTargetEncoder, fillna_cat, fillna_text

RANDOM_STATE = 42
N_FOLDS = 5

NUM_COLS = [
    "item_condition_id",
    "shipping",
    "category_depth",
    "name_length",
    "desc_length",
    "name_word_count",
    "has_description",
    "is_branded",
]
CAT_COLS = ["main_category", "sub_category", "sub_sub_category", "condition_label"]


def build_pipeline() -> Pipeline:
    """Reconstructs the exact architecture from notebooks/pipeline.ipynb."""
    name_pipe = Pipeline(
        [
            ("fillna", FunctionTransformer(fillna_text)),
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
        ]
    )
    desc_pipe = Pipeline(
        [
            ("fillna", FunctionTransformer(fillna_text)),
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("fillna", FunctionTransformer(fillna_cat)),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("name_tfidf", name_pipe, "name"),
            ("desc_tfidf", desc_pipe, "item_description"),
            ("num", StandardScaler(), NUM_COLS),
            ("cat", cat_pipe, CAT_COLS),
            ("cat_enc", LeakSafeTargetEncoder("main_category"), ["main_category", "price"]),
            ("brand_enc", LeakSafeTargetEncoder("brand_name"), ["brand_name", "price"]),
        ]
    )
    return Pipeline(
        [("preprocess", preprocessor), ("model", Ridge(alpha=5.0, random_state=RANDOM_STATE))]
    )


def main() -> None:
    data_path = Path("data/processed/train.csv")
    if not data_path.exists():
        raise SystemExit(
            f"{data_path} not found. Run src/data_prep.py against the raw Kaggle "
            "dataset first (see README > Getting Started)."
        )

    df = pd.read_csv(data_path)
    y = df["log_price"]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df), start=1):
        t0 = time.time()
        fold_train, fold_val = df.iloc[train_idx], df.iloc[val_idx]

        pipeline = build_pipeline()
        pipeline.fit(fold_train, y.iloc[train_idx])
        pred = pipeline.predict(fold_val)

        rmsle = float(np.sqrt(mean_squared_error(y.iloc[val_idx], pred)))
        fold_scores.append(rmsle)
        print(f"[fold {fold_idx}/{N_FOLDS}] RMSLE={rmsle:.4f}  ({time.time() - t0:.1f}s)")

    fold_scores = np.array(fold_scores)
    print(f"\nFinal Ridge+text pipeline: {fold_scores.mean():.4f} +/- {fold_scores.std():.4f}")
    print(
        "Paste this into the README's model-comparison table alongside the tree-model CV numbers."
    )


if __name__ == "__main__":
    main()
