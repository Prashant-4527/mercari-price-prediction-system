import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.feature_engineering import engineer_raw_features
from src.interval_utils import get_prediction_interval

# NOTE: LeakSafeTargetEncoder (src/pipeline_components.py) is never referenced
# directly here, but must remain importable from that exact module path --
# joblib resolves pickled custom transformers by (module, qualname).


st.set_page_config(page_title="Mercari Price Prediction", page_icon="🛍️")


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("models/mercari_pipeline_final.joblib")
    interval_lookup = joblib.load("models/interval_lookup.joblib")
    return pipeline, interval_lookup


pipeline, interval_lookup = load_artifacts()

st.title("Mercari Price Prediction")
st.write(
    "Enter listing details to get an AI-suggested price, trained on real Mercari marketplace data."
)

name = st.text_input("Item Name", placeholder="e.g. Chanel Classic Flag Bag")
item_condition_id = st.slider("Condition (1=New, 5=Poor)", 1, 5, 3)
category_name = st.text_input("Category", placeholder="e.g. Women/Bags/Shoulder Bag")
brand_name = st.text_input("Brand (optional)", placeholder="e.g. Chanel")
shipping = st.radio("Shipping paid by seller?", [0, 1], horizontal=True)
item_description = st.text_area("Description (optional)")

if st.button("Predict Price"):
    raw_df = pd.DataFrame(
        [
            {
                "name": name,
                "item_condition_id": item_condition_id,
                "category_name": category_name,
                "brand_name": brand_name if brand_name else None,
                "shipping": shipping,
                "item_description": item_description if item_description else None,
            }
        ]
    )
    raw_df["price"] = 0

    engineered_df = engineer_raw_features(raw_df)
    pred_log = pipeline.predict(engineered_df)[0]
    pred_price = float(np.expm1(pred_log))
    lower, upper = get_prediction_interval(
        pred_log, interval_lookup["bin_edges"], interval_lookup["bin_bounds"]
    )

    col1, col2 = st.columns(2)
    col1.metric("Predicted Price", f"${pred_price:.2f}")
    col2.metric("Likely Range", f"${lower:.2f} - ${upper:.2f}")
