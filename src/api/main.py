from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

from .pydantic_models import CustomerFeatures, PredictionResponse

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts customer risk probability using a WoE-transformed model",
    version="1.0"
)

# =====================================================
# Load model from MLflow Registry
# =====================================================
MODEL_NAME = "credit-risk-model"
MODEL_STAGE = "Production"

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)

# =====================================================
# Health Check
# =====================================================
@app.get("/")
def health():
    return {"status": "API running"}

# =====================================================
# Prediction Endpoint
# =====================================================
@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):

    input_df = pd.DataFrame([features.dict()])

    prediction = model.predict(input_df)

    return PredictionResponse(
        risk_probability=float(prediction[0])
    )
