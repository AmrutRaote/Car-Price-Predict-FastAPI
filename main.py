from fastapi import FastAPI
from schema import CarFeatures, PredictionResponse
from model import predict_price, load_artifacts
from fastapi.responses import JSONResponse


# Setting up the FastAPI app
app = FastAPI(
    title="Car Price Prediction API",
    version="1.0"
)


# This runs once when the server starts.
# We load the ML model and feature columns into memory here
# so we don't have to reload them on every single request.
@app.on_event("startup")
def startup_event():
    load_artifacts()


# Simple health-check route — just to verify the server is running
@app.get("/")
def test():
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "Welcome to Car Price Predition API"
        }
    )


# Main prediction endpoint
# Accepts car details as JSON, runs them through the model,
# and returns the predicted selling price in lakhs
@app.post("/predict", response_model=PredictionResponse)
def predict(features: CarFeatures):
    # .model_dump() converts the Pydantic object to a plain dict
    price = predict_price(features.model_dump())
    return PredictionResponse(prediction_price=price)
