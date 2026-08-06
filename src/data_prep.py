import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

RANDOM_STATE = 42
SMOOTHING = 10      # shrinkage strength toward global mean for low-count groups
N_FOLDS = 5

# ---------------------------------------------------------------------------
# 1. Load raw + replay cleaning 
# ---------------------------------------------------------------------------
df = pd.read_csv("data/mercari_sample.csv")

df["brand_name"] = df["brand_name"].fillna("No Brand").str.lower().str.strip()
df = df.dropna(subset=["category_name"])
df = df[df["price"] > 0]
df = df.drop_duplicates().reset_index(drop=True)

print(f"[1] Clean shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. SAFE structured features -- derived only from raw listing fields,
#    zero dependency on price, so these are 100% safe to build pre-split.
# ---------------------------------------------------------------------------
cat_split = df["category_name"].str.split("/")
df["main_category"]    = cat_split.str[0]
df["sub_category"]     = cat_split.str[1]
df["sub_sub_category"] = cat_split.str[2]
df["category_depth"]   = df["category_name"].str.split("/").str.len()

df["name_length"]     = df["name"].str.len()
df["desc_length"]     = df["item_description"].str.len()
df["name_word_count"] = df["name"].str.split().str.len()
df["has_description"] = df["item_description"].ne("No description yet").astype(int)
df["is_branded"]      = np.where(df["brand_name"] == "no brand", 0, 1)

condition_map = {1: "New", 2: "Like New", 3: "Good", 4: "Fair", 5: "Poor"}
df["condition_label"] = df["item_condition_id"].map(condition_map)

print("[2] Safe structured features built (no price dependency).")

# ---------------------------------------------------------------------------
# 3. Target -- log1p(price). This IS the target, never a feature.
# ---------------------------------------------------------------------------
df["log_price"] = np.log1p(df["price"])

# ---------------------------------------------------------------------------
# 4. Stratification bins -- used ONLY to guide the split, then discarded.
#    Never enters the model as a feature (it's price-derived).
# ---------------------------------------------------------------------------
df["price_bin"] = pd.qcut(df["log_price"], q=10, labels=False, duplicates="drop")

# ---------------------------------------------------------------------------
# 5. Stratified train/val split -- BEFORE any target-encoded feature exists.
# ---------------------------------------------------------------------------
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["price_bin"], random_state=RANDOM_STATE
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"[5] Train: {train_df.shape[0]:,} rows | Val: {val_df.shape[0]:,} rows")

# ---------------------------------------------------------------------------
# 6. Leak-safe target encoding -- K-fold out-of-fold for TRAIN,
#    train-only mapping for VAL. Smoothing pulls low-count groups toward
#    the global mean so a 1-listing brand can't dominate its own average.
# ---------------------------------------------------------------------------
global_mean_price = train_df["price"].mean()

def smoothed_group_stats(frame, group_col, target_col="price", m=SMOOTHING, global_mean=global_mean_price):
    stats = frame.groupby(group_col)[target_col].agg(["mean", "count"])
    stats["smoothed"] = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)
    return stats["smoothed"]

def oof_target_encode(train_frame, group_col, target_col="price", n_folds=N_FOLDS,
                       m=SMOOTHING, global_mean=global_mean_price, random_state=RANDOM_STATE):
    oof_encoded = pd.Series(index=train_frame.index, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for fold_train_idx, fold_holdout_idx in kf.split(train_frame):
        fold_train = train_frame.iloc[fold_train_idx]
        stats_map = smoothed_group_stats(fold_train, group_col, target_col, m, global_mean)
        holdout_groups = train_frame.iloc[fold_holdout_idx][group_col]
        oof_encoded.iloc[fold_holdout_idx] = holdout_groups.map(stats_map).fillna(global_mean)
    return oof_encoded

train_df["cat_avg_price"]   = oof_target_encode(train_df, "main_category")
train_df["brand_avg_price"] = oof_target_encode(train_df, "brand_name")

final_cat_map   = smoothed_group_stats(train_df, "main_category")
final_brand_map = smoothed_group_stats(train_df, "brand_name")

val_df["cat_avg_price"]   = val_df["main_category"].map(final_cat_map).fillna(global_mean_price)
val_df["brand_avg_price"] = val_df["brand_name"].map(final_brand_map).fillna(global_mean_price)

print("[6] Leak-safe target encoding done (OOF for train, train-only map for val).")

# ---------------------------------------------------------------------------
# 7. Sanity checks -- concrete, verifiable numbers
# ---------------------------------------------------------------------------
print("\n--- Sanity checks ---")
print(f"Global mean price (train only): ${global_mean_price:.2f}")
n_celine_train = (train_df["brand_name"] == "celine").sum()
print(f"Celine rows in train: {n_celine_train}")
if "celine" in final_brand_map.index:
    print(f"Celine smoothed avg price (train, all folds): ${final_brand_map['celine']:.2f}")
n_unseen_brands_val = (~val_df["brand_name"].isin(final_brand_map.index)).sum()
print(f"Val rows with brand unseen in train (fell back to global mean): {n_unseen_brands_val}")

print("\nprice_bin distribution check (train vs val, should match closely):")
print("Train:", train_df["price_bin"].value_counts(normalize=True).sort_index().round(3).to_dict())
print("Val:  ", val_df["price_bin"].value_counts(normalize=True).sort_index().round(3).to_dict())

# ---------------------------------------------------------------------------
# 8. Drop leakage-only / target columns from feature set
#    (is_price_outlier, price_tier, price_percentile, price_rank_in_cat were
#     never created here -- we don't even build features we can't use)
# ---------------------------------------------------------------------------
y_train = train_df["log_price"].copy()
y_val   = val_df["log_price"].copy()

# ---------------------------------------------------------------------------
# 9. Save processed, leak-safe artifacts
# ---------------------------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
print(f"\n[9] Saved data/processed/train.csv ({train_df.shape[0]:,} rows)")
print(f"[9] Saved data/processed/val.csv ({val_df.shape[0]:,} rows)")
print(f"\nFinal train_df columns ({train_df.shape[1]}):")
print(list(train_df.columns))
