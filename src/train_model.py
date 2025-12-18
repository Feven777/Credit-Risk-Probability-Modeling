import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

import lightgbm as lgb
import mlflow
import mlflow.lightgbm

from data_processing import process_data

# -----------------------------
# Configuration
# -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
IV_THRESHOLD = 0.02
TARGET_COL = "FraudResult"

MLFLOW_EXPERIMENT_NAME = "credit-risk-lightgbm"


# -----------------------------
# Metric evaluation function
# -----------------------------
def evaluate_model(y_true, y_pred, y_proba):
    """Compute evaluation metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


# -----------------------------
# Threshold selection using max F1
# -----------------------------
def find_best_threshold(y_true, y_proba):
    """Find threshold that maximizes F1-score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_threshold, f1_scores[best_idx]


# -----------------------------
# Main training pipeline
# -----------------------------
def main():
    # 1️⃣ Load data
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
    df = pd.read_csv(data_path)

    # 2️⃣ Feature engineering & WoE processing
    X, y, iv_df = process_data(df, iv_threshold=IV_THRESHOLD)

    # 3️⃣ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 🔹 Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # 4️⃣ LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 5️⃣ Model hyperparameters
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos_weight,
        "seed": RANDOM_STATE,
        "verbose": -1,
    }

    # 6️⃣ MLflow experiment setup
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="LightGBM-MaxF1"):

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("iv_threshold", IV_THRESHOLD)
        mlflow.log_param("test_size", TEST_SIZE)

        # 7️⃣ Train LightGBM model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, test_data],
            valid_names=["train", "valid"],
            num_boost_round=300,
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=0),
            ],
        )

        # 8️⃣ Predict probabilities
        y_proba = model.predict(X_test)

        # 🔹 Threshold selection (max F1)
        best_threshold, best_f1 = find_best_threshold(y_test, y_proba)
        y_pred = (y_proba >= best_threshold).astype(int)

        # 9️⃣ Evaluate model
        metrics = evaluate_model(y_test, y_pred, y_proba)

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        mlflow.log_metric("best_iteration", model.best_iteration)
        mlflow.log_metric("best_f1_at_threshold", best_f1)
        mlflow.log_param("chosen_threshold", float(best_threshold))

        # 🔟 Log model artifact
        mlflow.lightgbm.log_model(
            model,
            name="model",
            registered_model_name="CreditRiskLightGBM",
        )

        # Save IV table for traceability
        iv_df_path = os.path.join(BASE_DIR, "iv_values.csv")
        iv_df.to_csv(iv_df_path, index=False)
        mlflow.log_artifact(iv_df_path)

        # 1️⃣1️⃣ Print results
        print("Training completed successfully.")
        print(f"Chosen threshold (max F1): {best_threshold:.4f}")
        print("Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
