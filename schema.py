from enum import Enum
from pydantic import BaseModel, Field


# --- Enums for categorical fields ---
# These restrict the API input to only valid values,
# so users can't pass random strings like "Electric" or "Both"

class FuelType(str, Enum):
    petrol = "Petrol"
    diesel = "Diesel"
    cng = "Cng"


class SellerType(str, Enum):
    dealer = "Dealer"
    individual = "Individual"


class TransmissionType(str, Enum):
    manual = "Manual"
    automatic = "Automatic"


# --- Request body schema ---
# This is what the /predict endpoint expects as JSON input.
# Field names must match the training dataset columns exactly.
class CarFeatures(BaseModel):
    Car_Name: str = Field(..., examples=["ritz"])
    Year: int = Field(..., examples=[2014])
    Present_Price: float = Field(..., examples=[5.59])
    Kms_Driven: int = Field(..., examples=[27000])

    Fuel_Type: FuelType
    Seller_Type: SellerType
    Transmission: TransmissionType

    # Owner count: 0 = first owner, 1 = second, 3 = third
    Owner: int = Field(
        ...,
        ge=0,
        le=3,
        examples=[0],
        description="Number of previous Owner (0, 1, 2 or 3)"
    )


# --- Response schema ---
# The API sends back just the predicted price
class PredictionResponse(BaseModel):
    prediction_price: float