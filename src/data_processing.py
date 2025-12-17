import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans



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

def calculate_woe_iv(df, feature, target):
    """
    Calculate WoE and IV for a single feature.
    """

    eps = 1e-6  # avoid division by zero

    grouped = (
        df.groupby(feature, observed=True)[target]
        .agg(["count", "sum"])
        .rename(columns={"sum": "bad"})
    )

    grouped["good"] = grouped["count"] - grouped["bad"]

    grouped["dist_good"] = grouped["good"] / grouped["good"].sum()
    grouped["dist_bad"] = grouped["bad"] / grouped["bad"].sum()

    grouped["woe"] = np.log(
        (grouped["dist_good"] + eps) / (grouped["dist_bad"] + eps)
    )

    grouped["iv"] = (
        (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]
    )

    iv = grouped["iv"].sum()

    return grouped[["woe"]], iv


def apply_woe_iv(df, target_col, numeric_cols):
    """
    Apply WoE transformation and compute IV.
    """

    df = df.copy()
    iv_list = {}

    for col in numeric_cols:
        df[col] = pd.qcut(df[col], q=10, duplicates="drop")
        woe_table, iv = calculate_woe_iv(df, col, target_col)
        df[col] = df[col].map(woe_table["woe"])
        iv_list[col] = iv

    iv_df = pd.DataFrame(
        iv_list.items(), columns=["Feature", "IV"]
    ).sort_values("IV", ascending=False)

    X = df[numeric_cols]
    y = df[target_col]

    return X, y, iv_df

IV_THRESHOLD = 0.02

def drop_low_iv_features(X: pd.DataFrame, iv_df: pd.DataFrame, threshold: float = IV_THRESHOLD):
    """
    Drop features from X whose IV is below threshold.
    """
    # Select features with IV >= threshold
    selected_features = iv_df[iv_df["IV"] >= threshold]["Feature"].tolist()
    
    # Keep only these features in X
    X_selected = X[selected_features].copy()
    
    return X_selected

def calculate_rfm(df: pd.DataFrame, snapshot_date: str = None) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, Monetary (RFM) features per customer.
    """
    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # Use max transaction date +1 if snapshot_date not provided
    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)
    else:
        snapshot_date = pd.to_datetime(snapshot_date)

    rfm = df.groupby("CustomerId").agg(
        Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionId", "count"),
        Monetary=("Amount", "sum")
    ).reset_index()

    return rfm

def scale_rfm(rfm_df: pd.DataFrame):
    """
    Standardize RFM features for clustering.
    """
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[["Recency", "Frequency", "Monetary"]])
    return rfm_scaled, scaler

def cluster_customers(rfm_scaled, n_clusters=3, random_state=42):
    """
    Cluster customers using KMeans.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(rfm_scaled)
    return cluster_labels, kmeans

def assign_high_risk_label(rfm_df: pd.DataFrame, cluster_labels):
    """
    Identify high-risk customers (least engaged cluster).
    """
    rfm_df = rfm_df.copy()
    rfm_df["Cluster"] = cluster_labels

    # Compute average RFM per cluster
    cluster_summary = rfm_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

    # High-risk cluster: low Frequency & low Monetary
    high_risk_cluster = cluster_summary.sort_values(
        by=["Frequency", "Monetary"], ascending=[True, True]
    ).index[0]

    # Create binary target
    rfm_df["is_high_risk"] = (rfm_df["Cluster"] == high_risk_cluster).astype(int)

    return rfm_df[["CustomerId", "is_high_risk"]]

def process_data(df: pd.DataFrame, iv_threshold: float = 0.02):
    """
    Full data processing pipeline:
    - Time features
    - Aggregate features
    - WoE transformation on numeric features
    - Drop low-IV features
    """

    df = extract_time_features(df)
    df = create_aggregate_features(df)

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

    NUMERIC_COLS = [
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

    # Apply WoE + IV
    X, y, iv_df = apply_woe_iv(df_model, target_col="FraudResult", numeric_cols=NUMERIC_COLS)

    # Drop low-IV features
    X_selected = drop_low_iv_features(X, iv_df, threshold=iv_threshold)

    return X_selected, y, iv_df





