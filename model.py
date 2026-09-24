from pathlib import Path

import joblib
import pandas as pd


# Figure out where this file lives so we can find the .pkl files next to it
CAR_PRICE_API_DIR = Path(__file__).resolve().parent

MODEL_PATH = CAR_PRICE_API_DIR / "random_forest_model.pkl"
COLS_PATH = CAR_PRICE_API_DIR / "feature_columns.pkl"

# These hold the loaded model and column list in memory.
# We use None initially and load them lazily (only once).
_model = None
_feature_columns = None


def load_artifacts():
    """Load the trained model and feature columns from disk (only if not already loaded)."""
    global _model, _feature_columns

    if _model is None:
        _model = joblib.load(MODEL_PATH)

    if _feature_columns is None:
        _feature_columns = joblib.load(COLS_PATH)


def preprocess(payload: dict) -> pd.DataFrame:
    """
    Takes the raw input dict from the API request and transforms it
    into a DataFrame that matches what the model was trained on.
    """
    df = pd.DataFrame([payload])

    # These are the columns we need to one-hot encode,
    # same as what we did during training in train.py
    categorical_cols = [
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Owner",
        "Car_Name"
    ]

    # Owner comes in as an int (0, 1, 3) from the API,
    # but pd.get_dummies() skips numeric columns by default.
    # Converting to string so it gets one-hot encoded properly.
    df["Owner"] = df["Owner"].astype(str)

    # One-hot encode all categorical columns (drop_first to avoid dummy trap)
    df_encoded = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    # The model expects the exact same columns it was trained on.
    # reindex() adds any missing columns (filled with 0) and drops extra ones,
    # so the input always matches the training shape.
    df_encoded = df_encoded.reindex(columns=_feature_columns, fill_value=0)

    return df_encoded


def predict_price(payload: dict) -> float:
    """Run the full pipeline: load model -> preprocess input -> predict."""
    load_artifacts()

    X = preprocess(payload)
    prediction = _model.predict(X)[0]

    return float(prediction)