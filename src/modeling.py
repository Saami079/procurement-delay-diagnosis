from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
FINAL_DIR = BASE_DIR / "data" / "final"
RESULTS_DIR = BASE_DIR / "outputs" / "results"
MODEL_DIR = BASE_DIR / "dashboard" / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    return pd.read_csv(FINAL_DIR / "final_request_level.csv")


def main() -> None:
    df = load_data()

    features = [
        "Priority",
        "Request_Type",
        "Department_Requesting",
        "Vendor_Type",
        "Request_Amount",
        "Complexity_Score",
        "System_Load",
        "Num_Stages",
        "Total_Processing",
        "Total_Waiting",
        "Max_Stage_Delay",
        "Is_High_Value_Request",
        "Is_High_Complexity",
    ]

    target = "Delayed_Flag"

    X = df[features].copy()
    y = df[target].copy()

    print("\nTarget distribution:")
    print(y.value_counts(normalize=True) * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    categorical = [
        "Priority",
        "Request_Type",
        "Department_Requesting",
        "Vendor_Type",
    ]

    numerical = [col for col in features if col not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numerical,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("\n=== MODEL PERFORMANCE ===\n")
    print(cm)
    print("\n")
    print(report)

    with open(RESULTS_DIR / "model_results.txt", "w", encoding="utf-8") as f:
        f.write("Features Used:\n")
        for feature in features:
            f.write(f"- {feature}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    joblib.dump(model, MODEL_DIR / "delay_classifier.joblib")
    joblib.dump(features, MODEL_DIR / "feature_columns.joblib")

    print("\nModel + features saved for dashboard.")


if __name__ == "__main__":
    main()