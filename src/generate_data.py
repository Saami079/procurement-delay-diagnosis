from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Keep generation reproducible.
np.random.seed(42)
random.seed(42)

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

NUM_REQUESTS = 5000

STAGES_MASTER = [
    ("Manager Approval", 1, "Manager", "Management"),
    ("Finance Approval", 2, "Finance Officer", "Finance"),
    ("Procurement Review", 3, "Procurement Officer", "Procurement"),
    ("Final Approval", 4, "Senior Manager", "Management"),
]

REQUEST_TYPES = ["IT Purchase", "Office Supplies", "Vendor Contract", "Equipment"]
PRIORITIES = ["Low", "Medium", "High"]
DEPARTMENTS = ["IT", "HR", "Operations", "Finance", "Admin"]
VENDOR_TYPES = ["Internal", "External"]

REWORK_PROBS = {
    "Manager Approval": 0.07,
    "Finance Approval": 0.15,
    "Procurement Review": 0.18,
    "Final Approval": 0.05,
}


def get_complexity(request_type: str, amount: float) -> int:
    """Assign a bounded complexity score using request type and amount."""
    base = {
        "Office Supplies": 1,
        "IT Purchase": 2,
        "Equipment": 3,
        "Vendor Contract": 4,
    }[request_type]

    if amount > 100000:
        base += 1

    return min(base, 5)


def get_sla(complexity: int, priority: str, request_type: str) -> float:
    """Set a more realistic SLA so the target is learnable and not one-sided."""
    type_base = {
        "Office Supplies": 96,
        "IT Purchase": 120,
        "Equipment": 144,
        "Vendor Contract": 168,
    }[request_type]

    sla = type_base + (complexity * 18)

    if priority == "High":
        sla *= 0.90
    elif priority == "Low":
        sla *= 1.10

    return round(sla, 2)


def adjust_to_business_hours(dt: datetime) -> datetime:
    """Keep timestamps within the simple business-hour window."""
    if dt.hour < 9:
        return dt.replace(hour=9, minute=0, second=0, microsecond=0)

    if dt.hour >= 18:
        return (dt + timedelta(days=1)).replace(
            hour=9,
            minute=0,
            second=0,
            microsecond=0,
        )

    return dt


def get_processing_time(
    stage: str,
    priority: str,
    complexity: int,
    vendor_type: str,
    amount: float,
) -> float:
    """Generate stage processing time with realistic but not extreme delay drivers."""
    base = {
        "Manager Approval": np.random.randint(4, 12),
        "Finance Approval": np.random.randint(12, 36),
        "Procurement Review": np.random.randint(10, 30),
        "Final Approval": np.random.randint(4, 12),
    }[stage]

    base *= 1 + complexity * 0.10

    if priority == "High":
        base *= 0.80
    elif priority == "Low":
        base *= 1.15

    if vendor_type == "External":
        base *= 1.10

    if amount > 150000:
        base *= 1.10

    if np.random.rand() < 0.08:
        base *= np.random.uniform(1.25, 1.75)

    return round(base, 2)


def get_waiting_time(
    priority: str,
    complexity: int,
    stage: str,
    system_load: float,
) -> float:
    """Generate queue time before processing starts."""
    base = np.random.randint(1, 4)
    base *= 1 + complexity * 0.12

    if priority == "Low":
        base *= 1.25

    if stage == "Finance Approval":
        base *= 1.25

    base *= 1 + system_load * 0.20
    return round(base, 2)


def generate_stage_level_data(num_requests: int = NUM_REQUESTS) -> pd.DataFrame:
    """Create synthetic stage-level workflow records."""
    records: list[dict] = []
    base_date = datetime(2025, 1, 1, 9)

    for i in range(1, num_requests + 1):
        request_id = f"R{i:05d}"

        request_type = random.choice(REQUEST_TYPES)
        priority = random.choice(PRIORITIES)
        department = random.choice(DEPARTMENTS)
        vendor_type = random.choice(VENDOR_TYPES)

        amount = int(np.random.randint(1000, 250000))
        complexity = get_complexity(request_type, amount)
        sla = get_sla(complexity, priority, request_type)

        request_start = base_date + timedelta(days=int(np.random.randint(0, 200)))
        request_start = adjust_to_business_hours(request_start)

        previous_end = request_start
        system_load = float(np.random.rand())

        stage_sequence = STAGES_MASTER.copy()

        # Simpler low-value office supplies may skip the final approval.
        if amount < 8000 and request_type == "Office Supplies":
            stage_sequence = STAGES_MASTER[:-1]

        stage_index = 0
        rework_count = 0
        max_rework = 3

        while stage_index < len(stage_sequence):
            stage_name, order, role, dept = stage_sequence[stage_index]

            waiting = get_waiting_time(priority, complexity, stage_name, system_load)
            stage_start = adjust_to_business_hours(
                previous_end + timedelta(hours=waiting)
            )

            processing = get_processing_time(
                stage_name,
                priority,
                complexity,
                vendor_type,
                amount,
            )
            stage_end = adjust_to_business_hours(
                stage_start + timedelta(hours=processing)
            )

            records.append(
                {
                    "Request_ID": request_id,
                    "Stage": stage_name,
                    "Stage_Order": order,
                    "Role": role,
                    "Department_Stage": dept,
                    "Department_Requesting": department,
                    "Request_Type": request_type,
                    "Priority": priority,
                    "Vendor_Type": vendor_type,
                    "Request_Amount": amount,
                    "Complexity_Score": complexity,
                    "SLA_Hours": sla,
                    "System_Load": system_load,
                    "Start_Time": stage_start,
                    "End_Time": stage_end,
                    "Processing_Time": processing,
                    "Waiting_Time": waiting,
                }
            )

            previous_end = stage_end

            # Rework loops make the process more realistic and create bottlenecks.
            if (
                np.random.rand() < REWORK_PROBS[stage_name]
                and rework_count < max_rework
            ):
                rework_count += 1
                jump_back = random.choice([1, 1, 2])
                stage_index = max(0, stage_index - jump_back)
                previous_end = adjust_to_business_hours(
                    previous_end + timedelta(hours=4)
                )

                records.append(
                    {
                        "Request_ID": request_id,
                        "Stage": f"Rework to {stage_sequence[stage_index][0]}",
                        "Stage_Order": order,
                        "Role": role,
                        "Department_Stage": dept,
                        "Department_Requesting": department,
                        "Request_Type": request_type,
                        "Priority": priority,
                        "Vendor_Type": vendor_type,
                        "Request_Amount": amount,
                        "Complexity_Score": complexity,
                        "SLA_Hours": sla,
                        "System_Load": system_load,
                        "Start_Time": previous_end,
                        "End_Time": previous_end,
                        "Processing_Time": 0,
                        "Waiting_Time": 4,
                    }
                )
            else:
                stage_index += 1

    return pd.DataFrame(records)


def create_request_level_data(stage_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stage-level data into request-level analytical features."""
    grouped = stage_df.groupby("Request_ID")

    request_df = grouped.agg(
        Request_Start=("Start_Time", "min"),
        Request_End=("End_Time", "max"),
        Total_Processing=("Processing_Time", "sum"),
        Total_Waiting=("Waiting_Time", "sum"),
        SLA_Hours=("SLA_Hours", "first"),
        Request_Type=("Request_Type", "first"),
        Priority=("Priority", "first"),
        Department_Requesting=("Department_Requesting", "first"),
        Vendor_Type=("Vendor_Type", "first"),
        Request_Amount=("Request_Amount", "first"),
        Complexity_Score=("Complexity_Score", "first"),
        System_Load=("System_Load", "first"),
        Num_Stages=("Stage", "count"),
    ).reset_index()

    request_df["Request_Start"] = pd.to_datetime(request_df["Request_Start"])
    request_df["Request_End"] = pd.to_datetime(request_df["Request_End"])

    request_df["Total_TAT"] = (
        request_df["Request_End"] - request_df["Request_Start"]
    ).dt.total_seconds() / 3600

    request_df["Delay_Ratio"] = request_df["Total_TAT"] / request_df["SLA_Hours"]
    request_df["SLA_Breach_Hours"] = (
        request_df["Total_TAT"] - request_df["SLA_Hours"]
    ).clip(lower=0)
    request_df["Delayed_Flag"] = (
        request_df["Total_TAT"] > request_df["SLA_Hours"]
    ).astype(int)
    request_df["Is_High_Value_Request"] = (
        request_df["Request_Amount"] > 150000
    ).astype(int)
    request_df["Is_High_Complexity"] = (request_df["Complexity_Score"] >= 4).astype(int)

    stage_delay = stage_df.copy()
    stage_delay["Total_Stage_Time"] = (
        stage_delay["Processing_Time"] + stage_delay["Waiting_Time"]
    )

    bottleneck_idx = stage_delay.groupby("Request_ID")["Total_Stage_Time"].idxmax()
    bottleneck_df = stage_delay.loc[
        bottleneck_idx, ["Request_ID", "Stage", "Total_Stage_Time"]
    ].copy()
    bottleneck_df.columns = ["Request_ID", "Bottleneck_Stage", "Max_Stage_Delay"]

    request_df = request_df.merge(bottleneck_df, on="Request_ID", how="left")

    return request_df


def save_raw_data(stage_df: pd.DataFrame, request_df: pd.DataFrame) -> None:
    """Persist raw synthetic outputs for the rest of the pipeline."""
    stage_path = RAW_DIR / "stage_level_raw.csv"
    request_path = RAW_DIR / "request_level_raw.csv"

    stage_df.to_csv(stage_path, index=False)
    request_df.to_csv(request_path, index=False)

    print(f"Stage-level raw data saved to: {stage_path}")
    print(f"Request-level raw data saved to: {request_path}")


def main() -> None:
    """Run full synthetic data creation."""
    stage_df = generate_stage_level_data()
    request_df = create_request_level_data(stage_df)
    save_raw_data(stage_df, request_df)

    print("\nSynthetic data generation completed.")
    print(f"Stage-level shape: {stage_df.shape}")
    print(f"Request-level shape: {request_df.shape}")


if __name__ == "__main__":
    main()
