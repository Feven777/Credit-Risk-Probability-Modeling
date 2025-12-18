from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    Amount: float
    Value: float
    transaction_hour: float
    transaction_day: float
    transaction_month: float
    transaction_year: float
    total_transaction_amount: float
    avg_transaction_amount: float
    transaction_count: float
    std_transaction_amount: float


class PredictionResponse(BaseModel):
    risk_probability: float
