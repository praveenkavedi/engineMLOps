# for data manipulation
import pandas as pd
import os
# for train/test split
from sklearn.model_selection import train_test_split
# for hugging face authentication to upload files
from huggingface_hub import HfApi, hf_hub_download

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_REPO_ID = "Praveen-kavedi/engine-predictive-maintenance"
DATASET_FILENAME = "engine_data.csv"

# Download the main dataset to a local path before reading with pandas
local_engine_dataset_path = hf_hub_download(
    repo_id=DATASET_REPO_ID,
    filename=DATASET_FILENAME,
    repo_type="dataset"
)
engine_dataset = pd.read_csv(local_engine_dataset_path)
print("Dataset loaded successfully.")

# Rename columns to remove spaces
RENAME_MAP = {
    "Engine rpm": "Engine_RPM",
    "Lub oil pressure": "Lub_Oil_Pressure",
    "Fuel pressure": "Fuel_Pressure",
    "Coolant pressure": "Coolant_Pressure",
    "lub oil temp": "Lub_Oil_Temperature",
    "Coolant temp": "Coolant_Temperature",
    "Engine Condition": "Engine_Condition",
}
engine_dataset = engine_dataset.rename(columns=RENAME_MAP)
print("Columns renamed successfully.")

# Define the target variable
TARGET_COL = "Engine_Condition"

# Define feature columns
FEATURE_COLS = [
    "Engine_RPM",           # Engine rotations per minute
    "Lub_Oil_Pressure",     # Lubricating oil pressure in bar
    "Fuel_Pressure",        # Fuel pressure in bar
    "Coolant_Pressure",     # Coolant pressure in bar
    "Lub_Oil_Temperature",  # Lubricating oil temperature in Celsius
    "Coolant_Temperature",  # Coolant temperature in Celsius
]

# Define predictor matrix (X) and target (y)
X = engine_dataset[FEATURE_COLS]
y = engine_dataset[TARGET_COL]

# Split the dataset into training and test sets (stratified 80/20)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,      # 20% of the data is reserved for testing
    random_state=42,    # Ensures reproducibility
    stratify=y          # Maintains class distribution in both splits
)

print(f"Train size: {Xtrain.shape}, Test size: {Xtest.shape}")

# Save splits to CSV
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Split files saved locally.")

# Upload split files back to HF dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
    )
    print(f"Uploaded: {file_path}")

print("All split files uploaded to Hugging Face successfully.")
