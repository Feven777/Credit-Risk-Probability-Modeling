import pandas as pd
from src.data_processing import (
    process_data,
    calculate_rfm, scale_rfm, cluster_customers, assign_high_risk_label, integrate_high_risk_target  # Task 4
)

# Load data
df = pd.read_csv("data/processed/processed_data.csv")

# Process data
X, y, iv_df = process_data(df)

print("Selected features:", X.columns.tolist())
print(iv_df.sort_values("IV", ascending=False))

rfm_df = calculate_rfm(df)
rfm_scaled, scaler = scale_rfm(rfm_df)
cluster_labels, kmeans_model = cluster_customers(rfm_scaled)
high_risk_df = assign_high_risk_label(rfm_df, cluster_labels)
df_with_target = integrate_high_risk_target(df, high_risk_df)

df_with_target.to_csv("data/processed/processed_data_with_target.csv", index=False)

print("Processed dataset saved with 'is_high_risk' target variable!")
print(df_with_target[["CustomerId", "is_high_risk"]].head())