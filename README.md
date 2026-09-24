# Car Price Prediction API

A REST API built with FastAPI that predicts the selling price of used cars based on features like car name, year, mileage, fuel type, etc. It uses a Random Forest model trained on the CarDekho dataset.

A Streamlit frontend is also included for easy interaction.

## Dataset

The model is trained on the [CarDekho dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) (`cardekho_data (1).csv`) which contains 301 records with the following columns:

| Column        | Description                               |
| ------------- | ----------------------------------------- |
| Car_Name      | Name of the car                           |
| Year          | Year the car was purchased                |
| Selling_Price | Price the owner wants to sell at (target) |
| Present_Price | Current ex-showroom price of the car      |
| Kms_Driven    | Kilometers driven                         |
| Fuel_Type     | Petrol, Diesel, or CNG                    |
| Seller_Type   | Dealer or Individual                      |
| Transmission  | Manual or Automatic                       |
| Owner         | Number of previous owners (0, 1, or 3)    |

## Project Structure

```
car-price-api/
    main.py                  # FastAPI app with routes
    model.py                 # Model loading, preprocessing, prediction
    schema.py                # Pydantic request/response schemas
    train.py                 # Script to train and save the model
    streamlit_app.py         # Streamlit frontend UI
    random_forest_model.pkl  # Trained Random Forest model
    feature_columns.pkl      # Column order used during training
    cardekho_data (1).csv    # Training dataset
    requirements.txt         # Python dependencies
    runtime.txt              # Python version for deployment
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd car-price-api
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 3. Train the model (optional, .pkl files are already included)

```bash
python train.py
```

This reads the CSV, trains a Random Forest regressor, and saves two files:

- `random_forest_model.pkl` -- the trained model
- `feature_columns.pkl` -- the list of feature columns after one-hot encoding

## Running the API

```bash
uvicorn main:app --reload
```

The server starts at `http://127.0.0.1:8000`.

- `GET /` -- Health check
- `POST /predict` -- Predict car selling price

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Car_Name": "swift",
    "Year": 2014,
    "Present_Price": 5.59,
    "Kms_Driven": 27000,
    "Fuel_Type": "Petrol",
    "Seller_Type": "Dealer",
    "Transmission": "Manual",
    "Owner": 0
  }'
```

### Example Response

```json
{
  "prediction_price": 3.8
}
```

### API Docs

FastAPI auto-generates interactive docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Running the Streamlit Frontend

Make sure the FastAPI server is running first, then in a separate terminal:

```bash
streamlit run streamlit_app.py
```

This opens a web UI where you can fill in car details and get the predicted price.

## Deployment

The API is deployed on Render. To deploy your own:

1. Push your code to GitHub.
2. Create a new Web Service on [Render](https://render.com).
3. Set the build command to `pip install -r requirements.txt`.
4. Set the start command to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
5. Update `API_URL` in `streamlit_app.py` to your deployed URL.

## Tech Stack

- Python
- FastAPI
- scikit-learn (Random Forest)
- pandas
- Streamlit
- Render (deployment)
