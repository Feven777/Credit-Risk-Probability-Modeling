import pandas as pd
import numpy as np
import pytest

from src.data_processing import (
    extract_time_features,
    create_aggregate_features,
    calculate_rfm,
    process_data,
)

@pytest.fixture
def sample_transactions():
    """
    Small synthetic dataset for testing.
    """
    return pd.DataFrame({
        "TransactionId": [1, 2, 3, 4],
        "CustomerId": [101, 101, 102, 102],
        "TransactionStartTime": [
            "2024-01-01 10:00:00",
            "2024-01-02 12:00:00",
            "2024-01-01 09:00:00",
            "2024-01-03 14:00:00",
        ],
        "Amount": [100, 200, 150, 300],
        "Value": [100, 200, 150, 300],
        "TransactionId": [1, 2, 3, 4],
        "BatchId": [1, 1, 2, 2],
        "AccountId": [10, 10, 20, 20],
        "SubscriptionId": [5, 5, 6, 6],
        "CountryCode": [256, 256, 256, 256],
        "ProductCategory": ["A", "A", "B", "B"],
        "ChannelId": ["Web", "Web", "Mobile", "Mobile"],
        "ProviderId": ["P1", "P1", "P2", "P2"],
        "PricingStrategy": [1, 1, 2, 2],
        "FraudResult": [0, 1, 0, 1],
    })


# --------------------------------------------------
# Test 1: Time feature extraction
# --------------------------------------------------

def test_extract_time_features_creates_columns(sample_transactions):
    df = extract_time_features(sample_transactions)

    expected_columns = {
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
    }

    assert expected_columns.issubset(df.columns)


# --------------------------------------------------
# Test 2: Aggregate feature creation
# --------------------------------------------------

def test_create_aggregate_features(sample_transactions):
    df = create_aggregate_features(sample_transactions)

    # Check aggregate columns exist
    assert "total_transaction_amount" in df.columns
    assert "avg_transaction_amount" in df.columns
    assert "transaction_count" in df.columns

    # Validate aggregation logic
    customer_101 = df[df["CustomerId"] == 101].iloc[0]
    assert customer_101["total_transaction_amount"] == 300
    assert customer_101["transaction_count"] == 2


# --------------------------------------------------
# Test 3: RFM calculation
# --------------------------------------------------

def test_calculate_rfm(sample_transactions):
    rfm = calculate_rfm(sample_transactions)

    assert set(["Recency", "Frequency", "Monetary"]).issubset(rfm.columns)
    assert rfm.shape[0] == 2  # two customers


# --------------------------------------------------
# Test 4: Full processing pipeline
# --------------------------------------------------

def test_process_data_output_shapes(sample_transactions):
    X, y, iv_df = process_data(sample_transactions, iv_threshold=0.0)

    # X and y should align
    assert len(X) == len(y)

    # Target should be binary
    assert set(y.unique()).issubset({0, 1})

    # IV dataframe structure
    assert set(["Feature", "IV"]).issubset(iv_df.columns)
