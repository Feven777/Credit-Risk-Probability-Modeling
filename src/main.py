import pandas as pd
from src.data_processing import process_data

# Load data
df = pd.read_csv("data/processed/processed_data.csv")

# Process data
X, y, iv_df = process_data(df)

print("Selected features:", X.columns.tolist())
print(iv_df.sort_values("IV", ascending=False))
