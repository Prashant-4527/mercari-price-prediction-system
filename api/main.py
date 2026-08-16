import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.feature_engineering import engineer_raw_features
from src.interval_utils import get_prediction_interval

# NOTE: LeakSafeTargetEncoder (src/pipeline_components.py) is never referenced
# directly here, but must remain importable from that exact module path --
# joblib resolves pickled custom transformers by (module, qualname), not by
# whatever happens to be imported in this file's namespace.

app = FastAPI(title="Mercari Price Prediction API")

# Loaded ONCE at server startup, not per-request -- reused for every prediction
pipeline = joblib.load("models/mercari_pipeline_final.joblib")
interval_lookup = joblib.load("models/interval_lookup.joblib")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


class ListingInput(BaseModel):
    name: str = Field(..., min_length=1, description="Listing title")
    item_condition_id: int = Field(..., ge=1, le=5, description="1=New, 5=Poor")
    category_name: str = Field(..., min_length=1, description="e.g. Women/Bags/Shoulder Bag")
    brand_name: str | None = Field(default=None, description="Brand, if any")
    shipping: int = Field(..., ge=0, le=1, description="1 if seller pays shipping")
    item_description: str | None = Field(default=None, description="Listing description")


class PredictionResponse(BaseModel):
    predicted_price: float
    price_range_low: float
    price_range_high: float


@app.post("/predict", response_model=PredictionResponse)
def predict_price(listing: ListingInput) -> PredictionResponse:
    raw_df = pd.DataFrame([listing.model_dump()])
    raw_df["price"] = 0  # unused by transform(); see README Known Limitations #1

    engineered_df = engineer_raw_features(raw_df)
    pred_log = pipeline.predict(engineered_df)[0]
    pred_price = float(np.expm1(pred_log))

    lower, upper = get_prediction_interval(
        pred_log, interval_lookup["bin_edges"], interval_lookup["bin_bounds"]
    )

    return PredictionResponse(
        predicted_price=round(pred_price, 2),
        price_range_low=round(lower, 2),
        price_range_high=round(upper, 2),
    )
