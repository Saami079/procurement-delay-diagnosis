from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
FINAL_DIR = BASE_DIR / "data" / "final"
MODEL_DIR = BASE_DIR / "dashboard" / "models"


def load_stage_data():
    return pd.read_csv(FINAL_DIR / "final_stage_level.csv")


def load_request_data():
    return pd.read_csv(FINAL_DIR / "final_request_level.csv")


def load_model():
    model = joblib.load(MODEL_DIR / "delay_classifier.joblib")
    feature_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")
    return model, feature_cols
