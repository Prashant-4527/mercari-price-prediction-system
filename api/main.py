import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from src.feature_engineering import engineer_raw_features

app = FastAPI(title="Mercari Price Prediction API")

# Loaded ONCE at server startup, not per-request -- reused for every prediction
pipeline = joblib.load("models/mercari_pipeline_final.joblib")


@app.get("/health")
def health_check():
    return {"status": "ok"}


class ListingInput(BaseModel):
    name: str = Field(..., min_length=1, description="Listing title")
    item_condition_id: int = Field(..., ge=1, le=5, description="1=New, 5=Poor")
    category_name: str = Field(..., min_length=1, description="e.g. Women/Bags/Shoulder Bag")
    brand_name: Optional[str] = Field(default=None, description="Brand, if any")
    shipping: int = Field(..., ge=0, le=1, description="1 if seller pays shipping")
    item_description: Optional[str] = Field(default=None, description="Listing description")


@app.post("/predict")
def predict_price(listing: ListingInput):
    raw_df = pd.DataFrame([listing.model_dump()])
    raw_df["price"] = 0  # placeholder only -- transform() never reads this, only fit() did (Day 10)

    engineered_df = engineer_raw_features(raw_df)
    pred_log = pipeline.predict(engineered_df)[0]
    pred_price = float(np.expm1(pred_log))

    return {"predicted_price": round(pred_price, 2)}