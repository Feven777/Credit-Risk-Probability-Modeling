import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from xverse.transformer import WOE



def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer-level aggregate features.
    """
    agg_df = (
        df.groupby("CustomerId")
        .agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            transaction_count=("Amount", "count"),
            std_transaction_amount=("Amount", "std"),
        )
        .reset_index()
    )

    return df.merge(agg_df, on="CustomerId", how="left")


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from TransactionStartTime.
    """
    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year

    return df



def build_preprocessing_pipeline():
    """
    Build sklearn preprocessing pipeline.
    """

    numerical_features = [
        "Amount",
        "Value",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
    ]

    categorical_features = [
        "ProductCategory",
        "ChannelId",
        "ProviderId",
        "PricingStrategy",
    ]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor



def process_data(df: pd.DataFrame):
    """
    Full data processing pipeline:
    - Aggregate features
    - Time features
    - Preprocessing pipeline
    """

    df = extract_time_features(df)
    df = create_aggregate_features(df)

    # Drop non-informative / ID columns
    drop_cols = [
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        "CustomerId",
        "TransactionStartTime",
        "CountryCode",
    ]

    df_model = df.drop(columns=drop_cols)

    X = df_model.drop(columns=["FraudResult"])
    y = df_model["FraudResult"]

    preprocessor = build_preprocessing_pipeline()
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor
