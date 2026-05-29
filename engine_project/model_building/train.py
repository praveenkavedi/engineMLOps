# for data manipulation
import pandas as pd
import numpy as np
import os
import json
# for model training, tuning, and evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score
# for model serialization
import joblib
# for hugging face authentication to upload files
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from pathlib import Path

# Define constants
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_REPO_ID = "Praveen-kavedi/engine-predictive-maintenance"
MODEL_REPO_ID = "Praveen-kavedi/engine-maintenance-rf"

FEATURE_COLS = [
    "Engine_RPM",           # Engine rotations per minute
    "Lub_Oil_Pressure",     # Lubricating oil pressure in bar
    "Fuel_Pressure",        # Fuel pressure in bar
    "Coolant_Pressure",     # Coolant pressure in bar
    "Lub_Oil_Temperature",  # Lubricating oil temperature in Celsius
    "Coolant_Temperature",  # Coolant temperature in Celsius
]
TARGET_COL = "Engine_Condition"
RANDOM_STATE = 42

# Download train and test splits from Hugging Face
train_path = hf_hub_download(
    repo_id=DATASET_REPO_ID,
    filename="Xtrain.csv",
    repo_type="dataset"
)
test_path = hf_hub_download(
    repo_id=DATASET_REPO_ID,
    filename="Xtest.csv",
    repo_type="dataset"
)
ytrain_path = hf_hub_download(
    repo_id=DATASET_REPO_ID,
    filename="ytrain.csv",
    repo_type="dataset"
)
ytest_path = hf_hub_download(
    repo_id=DATASET_REPO_ID,
    filename="ytest.csv",
    repo_type="dataset"
)

Xtrain = pd.read_csv(train_path)
Xtest = pd.read_csv(test_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

print(f"Train size: {Xtrain.shape}, Test size: {Xtest.shape}")

# Define base RandomForest model
rf_model = RandomForestClassifier(random_state=RANDOM_STATE)

# Define hyperparameter grid
param_grid = {
    "n_estimators": [150, 200],
    "max_depth": [8, 12, None],
    "max_features": ["sqrt", 0.5],
    "class_weight": ["balanced", None],
}

# Hyperparameter tuning with GridSearchCV
f1_scorer = f1_score.__class__ if False else __import__('sklearn.metrics', fromlist=['make_scorer']).make_scorer(f1_score)
grid_search = GridSearchCV(
    rf_model,
    param_grid,
    scoring=f1_scorer,
    cv=5,
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(Xtrain, ytrain)

print("Best CV F1:", round(grid_search.best_score_, 4))
print("Best parameters:", grid_search.best_params_)

# Store the best model
best_model = grid_search.best_estimator_

# Make predictions on train and test
y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

# Generate classification reports
train_report = classification_report(ytrain, y_pred_train, output_dict=True)
test_report = classification_report(ytest, y_pred_test, output_dict=True)

print("\nTrain classification report:")
print(classification_report(ytrain, y_pred_train))

print("\nTest classification report:")
print(classification_report(ytest, y_pred_test))

# Save experiment log to CSV and JSON
experiment_log = pd.DataFrame([{
    "hf_dataset": DATASET_REPO_ID,
    "algorithm": "RandomForestClassifier",
    "scoring": "f1",
    "cv_folds": 5,
    "best_cv_f1": round(grid_search.best_score_, 6),
    "train_accuracy": train_report['accuracy'],
    "train_precision": train_report['1']['precision'],
    "train_recall": train_report['1']['recall'],
    "train_f1": train_report['1']['f1-score'],
    "test_accuracy": test_report['accuracy'],
    "test_precision": test_report['1']['precision'],
    "test_recall": test_report['1']['recall'],
    "test_f1": test_report['1']['f1-score'],
    **grid_search.best_params_,
}])

Path("models").mkdir(exist_ok=True)
log_csv_path = "models/experiment_log.csv"
log_json_path = "models/experiment_log.json"
experiment_log.to_csv(log_csv_path, index=False)
with open(log_json_path, "w") as f:
    json.dump(experiment_log.iloc[0].to_dict(), f, indent=2)

print(f"Experiment log saved: {log_csv_path}, {log_json_path}")

# Save best model locally
MODEL_PATH = "models/best_random_forest.joblib"
joblib.dump(best_model, MODEL_PATH)
print(f"Model saved locally: {MODEL_PATH}")

# Upload model to Hugging Face model repo
try:
    api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
    print(f"Model repo '{MODEL_REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model repo '{MODEL_REPO_ID}' not found. Creating...")
    create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False)
    print(f"Model repo '{MODEL_REPO_ID}' created.")

api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo="best_random_forest.joblib",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
)
print(f"Model uploaded to: https://huggingface.co/{MODEL_REPO_ID}")
