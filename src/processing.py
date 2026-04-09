from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw stage-level and request-level files."""
    stage_df = pd.read_csv(RAW_DIR / "stage_level_raw.csv")
    request_df = pd.read_csv(RAW_DIR / "request_level_raw.csv")
    return stage_df, request_df


def process_stage_data(stage_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich stage-level data."""
    df = stage_df.copy()

    df["Start_Time"] = pd.to_datetime(df["Start_Time"])
    df["End_Time"] = pd.to_datetime(df["End_Time"])

    df["Total_Stage_Time"] = df["Processing_Time"] + df["Waiting_Time"]
    df["Rework_Flag"] = df["Stage"].str.contains("Rework", case=False, na=False).astype(int)
    df["Is_High_Value_Request"] = (df["Request_Amount"] > 150000).astype(int)
    df["Is_High_Complexity"] = (df["Complexity_Score"] >= 4).astype(int)

    df["Processing_Time"] = df["Processing_Time"].clip(lower=0)
    df["Waiting_Time"] = df["Waiting_Time"].clip(lower=0)
    df["Total_Stage_Time"] = df["Total_Stage_Time"].clip(lower=0)

    return df


def process_request_data(request_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich request-level data."""
    df = request_df.copy()

    df["Request_Start"] = pd.to_datetime(df["Request_Start"])
    df["Request_End"] = pd.to_datetime(df["Request_End"])

    df["Total_TAT"] = (df["Request_End"] - df["Request_Start"]).dt.total_seconds() / 3600
    df["Delay_Ratio"] = df["Total_TAT"] / df["SLA_Hours"]
    df["SLA_Breach_Hours"] = (df["Total_TAT"] - df["SLA_Hours"]).clip(lower=0)
    df["Delayed_Flag"] = (df["Total_TAT"] > df["SLA_Hours"]).astype(int)

    df["Is_High_Value_Request"] = (df["Request_Amount"] > 150000).astype(int)
    df["Is_High_Complexity"] = (df["Complexity_Score"] >= 4).astype(int)

    numeric_cols = [
        "Total_TAT",
        "Total_Processing",
        "Total_Waiting",
        "SLA_Hours",
        "Delay_Ratio",
        "SLA_Breach_Hours",
        "Request_Amount",
        "Complexity_Score",
        "Num_Stages",
        "Max_Stage_Delay",
        "System_Load",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    non_negative_cols = [
        "Total_TAT",
        "Total_Processing",
        "Total_Waiting",
        "SLA_Hours",
        "SLA_Breach_Hours",
        "Request_Amount",
        "Complexity_Score",
        "Num_Stages",
        "Max_Stage_Delay",
        "System_Load",
    ]

    for col in non_negative_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df


def save_processed_data(stage_df: pd.DataFrame, request_df: pd.DataFrame) -> None:
    """Save processed datasets for the finalization step."""
    stage_path = PROCESSED_DIR / "stage_level_processed.csv"
    request_path = PROCESSED_DIR / "request_level_processed.csv"

    stage_df.to_csv(stage_path, index=False)
    request_df.to_csv(request_path, index=False)

    print(f"Stage-level processed data saved to: {stage_path}")
    print(f"Request-level processed data saved to: {request_path}")


def main() -> None:
    """Run the processing layer of the pipeline."""
    stage_df, request_df = load_data()
    stage_df = process_stage_data(stage_df)
    request_df = process_request_data(request_df)
    save_processed_data(stage_df, request_df)

    print("\nProcessing completed.")
    print(f"Stage-level shape: {stage_df.shape}")
    print(f"Request-level shape: {request_df.shape}")


if __name__ == "__main__":
    main()