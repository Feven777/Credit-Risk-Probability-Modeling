import os
import re
import mlflow.lightgbm

# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..")
)

MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")
OUTPUT_FILE = os.path.join(BASE_DIR, "pydantic_models.py")


# =====================================================
# Validate mlruns directory
# =====================================================
if not os.path.exists(MLRUNS_DIR):
    raise FileNotFoundError(f"MLflow directory not found at {MLRUNS_DIR}")


# =====================================================
# Find latest experiment
# =====================================================
experiment_dirs = [
    d for d in os.listdir(MLRUNS_DIR)
    if d.isdigit() and os.path.isdir(os.path.join(MLRUNS_DIR, d))
]

if not experiment_dirs:
    raise FileNotFoundError("No MLflow experiments found")

latest_experiment = sorted(experiment_dirs, key=int)[-1]
EXPERIMENT_DIR = os.path.join(MLRUNS_DIR, latest_experiment)

print(f"Using MLflow experiment ID: {latest_experiment}")


# =====================================================
# Find latest VALID run (UUID only)
# =====================================================
RUN_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")

run_dirs = [
    d for d in os.listdir(EXPERIMENT_DIR)
    if (
        os.path.isdir(os.path.join(EXPERIMENT_DIR, d))
        and RUN_ID_PATTERN.match(d)
    )
]

if not run_dirs:
    raise FileNotFoundError("No valid MLflow runs found")

latest_run = sorted(run_dirs)[-1]
RUN_DIR = os.path.join(EXPERIMENT_DIR, latest_run)


# =====================================================
# Locate model artifact dynamically
# =====================================================
ARTIFACTS_DIR = os.path.join(RUN_DIR, "models")


if not os.path.exists(ARTIFACTS_DIR):
    raise FileNotFoundError(f"No artifacts directory at {ARTIFACTS_DIR}")

# Find first directory containing MLmodel file
model_path = None

for item in os.listdir(ARTIFACTS_DIR):
    candidate = os.path.join(ARTIFACTS_DIR, item)
    if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "MLmodel")):
        model_path = candidate
        break

if model_path is None:
    raise FileNotFoundError(
        f"No MLflow model found inside artifacts directory:\n{ARTIFACTS_DIR}"
    )

print(f"Loading model from:\n{model_path}")


# =====================================================
# Load LightGBM model
# =====================================================
model = mlflow.lightgbm.load_model(model_path)
feature_names = model.booster_.feature_name()


# =====================================================
# Generate Pydantic models
# =====================================================
lines = [
    "from pydantic import BaseModel",
    "",
    "class CustomerData(BaseModel):"
]

for feature in feature_names:
    safe_feature = (
        feature
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )
    lines.append(f"    {safe_feature}: float")

lines.extend([
    "",
    "class PredictionResponse(BaseModel):",
    "    risk_probability: float"
])


# =====================================================
# Write output
# =====================================================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))


# =====================================================
# Done
# =====================================================
print("\n✅ Pydantic models generated successfully!")
print(f"📄 Saved at: {OUTPUT_FILE}")
print("\nGenerated code:\n")
print("\n".join(lines))
