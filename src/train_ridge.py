import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

train = pd.read_csv("data/processed/train.csv")
val   = pd.read_csv("data/processed/val.csv")
for c in ["name", "item_description"]:
    train[c] = train[c].fillna("")
    val[c]   = val[c].fillna("")

# Text features 
name_vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.9, stop_words="english")
desc_vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2, max_df=0.9, stop_words="english")
X_name_train = name_vec.fit_transform(train["name"]); X_name_val = name_vec.transform(val["name"])
X_desc_train = desc_vec.fit_transform(train["item_description"]); X_desc_val = desc_vec.transform(val["item_description"])

# Structured features 
NUM_COLS = ["item_condition_id","shipping","category_depth","name_length","desc_length",
            "name_word_count","has_description","is_branded","cat_avg_price","brand_avg_price"]
CAT_COLS = ["main_category","sub_category","sub_sub_category","condition_label"]
for c in CAT_COLS:
    train[c] = train[c].fillna("missing"); val[c] = val[c].fillna("missing")

scaler = StandardScaler()
X_num_train = scaler.fit_transform(train[NUM_COLS]); X_num_val = scaler.transform(val[NUM_COLS])

ohe = OneHotEncoder(handle_unknown="ignore")
X_cat_train = ohe.fit_transform(train[CAT_COLS]); X_cat_val = ohe.transform(val[CAT_COLS])

# Combine everything
X_train = sp.hstack([X_name_train, X_desc_train, sp.csr_matrix(X_num_train), X_cat_train]).tocsr()
X_val   = sp.hstack([X_name_val,   X_desc_val,   sp.csr_matrix(X_num_val),   X_cat_val]).tocsr()
y_train = train["log_price"].values
y_val   = val["log_price"].values

# Alpha sweep -
for alpha in [0.1, 1.0, 5.0, 20.0, 50.0]:
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    rmsle = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
    print(f"alpha={alpha:<6} val RMSLE={rmsle:.4f}")