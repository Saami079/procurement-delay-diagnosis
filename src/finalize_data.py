from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FINAL_DIR = BASE_DIR / "data" / "final"

FINAL_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed stage-level and request-level data."""
    stage_df = pd.read_csv(PROCESSED_DIR / "stage_level_processed.csv")
    request_df = pd.read_csv(PROCESSED_DIR / "request_level_processed.csv")
    return stage_df, request_df


def final_quality_checks(stage_df: pd.DataFrame, request_df: pd.DataFrame) -> None:
    """Print final sanity checks before saving final datasets."""
    print("\n--- FINAL DATA CHECKS ---")
    print("Stage-level shape:", stage_df.shape)
    print("Request-level shape:", request_df.shape)

    print("\nStage-level missing values:\n", stage_df.isnull().sum())
    print("\nRequest-level missing values:\n", request_df.isnull().sum())

    print("\nStage-level duplicates:", stage_df.duplicated().sum())
    print("Request-level duplicates:", request_df.duplicated().sum())


def save_final_data(stage_df: pd.DataFrame, request_df: pd.DataFrame) -> None:
    """Save final analytical datasets."""
    stage_path = FINAL_DIR / "final_stage_level.csv"
    request_path = FINAL_DIR / "final_request_level.csv"

    stage_df.to_csv(stage_path, index=False)
    request_df.to_csv(request_path, index=False)

    print("\nFinal datasets saved successfully.")
    print("Stage-level:", stage_path)
    print("Request-level:", request_path)


def main() -> None:
    """Run the final data export step."""
    stage_df, request_df = load_processed_data()
    final_quality_checks(stage_df, request_df)
    save_final_data(stage_df, request_df)


if __name__ == "__main__":
    main()