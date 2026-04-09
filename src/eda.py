from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
FINAL_DIR = BASE_DIR / "data" / "final"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
RESULTS_DIR = BASE_DIR / "outputs" / "results"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load final analytical datasets."""
    stage_df = pd.read_csv(FINAL_DIR / "final_stage_level.csv")
    request_df = pd.read_csv(FINAL_DIR / "final_request_level.csv")
    return stage_df, request_df


def save_summary_tables(
    stage_df: pd.DataFrame,
    request_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create and export key EDA summary tables."""
    stage_summary = (
        stage_df.groupby("Stage")[["Processing_Time", "Waiting_Time", "Total_Stage_Time"]]
        .mean()
        .sort_values("Total_Stage_Time", ascending=False)
        .round(2)
    )
    stage_summary.to_csv(RESULTS_DIR / "stage_delay_summary.csv")

    bottleneck_summary = request_df["Bottleneck_Stage"].value_counts().reset_index()
    bottleneck_summary.columns = ["Bottleneck_Stage", "Count"]
    bottleneck_summary.to_csv(RESULTS_DIR / "bottleneck_stage_summary.csv", index=False)

    request_summary = (
        request_df[
            [
                "Total_TAT",
                "Total_Processing",
                "Total_Waiting",
                "SLA_Hours",
                "Delay_Ratio",
                "SLA_Breach_Hours",
            ]
        ]
        .describe()
        .round(2)
    )
    request_summary.to_csv(RESULTS_DIR / "request_summary.csv")

    delayed_summary = request_df["Delayed_Flag"].value_counts().reset_index()
    delayed_summary.columns = ["Delayed_Flag", "Count"]
    delayed_summary.to_csv(RESULTS_DIR / "delayed_flag_summary.csv", index=False)

    rework_summary = stage_df["Rework_Flag"].value_counts().reset_index()
    rework_summary.columns = ["Rework_Flag", "Count"]
    rework_summary.to_csv(RESULTS_DIR / "rework_summary.csv", index=False)

    return stage_summary, request_summary, bottleneck_summary


def plot_bar(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Save a bar chart to the figures folder."""
    plt.figure(figsize=figsize)
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()


def plot_hist(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    bins: int = 30,
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """Save a histogram to the figures folder."""
    plt.figure(figsize=figsize)
    plt.hist(series.dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()


def plot_boxplot_by_category(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Save a category-wise boxplot."""
    categories = list(df[category_col].dropna().unique())
    data = [df[df[category_col] == cat][value_col].dropna() for cat in categories]

    plt.figure(figsize=figsize)
    plt.boxplot(data, tick_labels=categories)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """Save a scatter plot."""
    plt.figure(figsize=figsize)
    plt.scatter(df[x_col], df[y_col], alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()


def create_stage_level_visuals(stage_df: pd.DataFrame, request_df: pd.DataFrame) -> None:
    """Create core stage-level visuals."""
    avg_total_stage = (
        stage_df.groupby("Stage")["Total_Stage_Time"].mean().sort_values(ascending=False)
    )
    plot_bar(
        avg_total_stage,
        "Average Total Stage Time by Stage",
        "Stage",
        "Hours",
        "avg_total_stage_time.png",
    )

    avg_waiting = stage_df.groupby("Stage")["Waiting_Time"].mean().sort_values(ascending=False)
    plot_bar(
        avg_waiting,
        "Average Waiting Time by Stage",
        "Stage",
        "Hours",
        "avg_waiting_time_by_stage.png",
    )

    avg_processing = (
        stage_df.groupby("Stage")["Processing_Time"].mean().sort_values(ascending=False)
    )
    plot_bar(
        avg_processing,
        "Average Processing Time by Stage",
        "Stage",
        "Hours",
        "avg_processing_time_by_stage.png",
    )

    stage_counts = stage_df["Stage"].value_counts()
    plot_bar(
        stage_counts,
        "Stage Frequency",
        "Stage",
        "Count",
        "stage_frequency.png",
    )

    bottleneck_counts = request_df["Bottleneck_Stage"].value_counts()
    plot_bar(
        bottleneck_counts,
        "Bottleneck Stage Frequency",
        "Stage",
        "Count",
        "bottleneck_stage_frequency.png",
    )


def create_request_level_visuals(request_df: pd.DataFrame) -> None:
    """Create core request-level visuals."""
    plot_hist(
        request_df["Total_TAT"],
        "Distribution of Total Turnaround Time",
        "Total TAT (Hours)",
        "Frequency",
        "total_tat_distribution.png",
    )

    delayed_counts = request_df["Delayed_Flag"].value_counts().sort_index()
    plot_bar(
        delayed_counts,
        "Delayed vs Not Delayed",
        "Delayed_Flag",
        "Count",
        "delayed_flag_distribution.png",
        figsize=(6, 4),
    )

    avg_tat_by_type = (
        request_df.groupby("Request_Type")["Total_TAT"].mean().sort_values(ascending=False)
    )
    plot_bar(
        avg_tat_by_type,
        "Average Total TAT by Request Type",
        "Request Type",
        "Hours",
        "avg_tat_by_request_type.png",
    )

    avg_tat_by_department = (
        request_df.groupby("Department_Requesting")["Total_TAT"]
        .mean()
        .sort_values(ascending=False)
    )
    plot_bar(
        avg_tat_by_department,
        "Average Total TAT by Department",
        "Department",
        "Hours",
        "avg_tat_by_department.png",
    )

    plot_boxplot_by_category(
        request_df,
        "Priority",
        "Total_TAT",
        "Total TAT by Priority",
        "Priority",
        "Hours",
        "tat_by_priority_boxplot.png",
    )

    plot_scatter(
        request_df,
        "Request_Amount",
        "Total_TAT",
        "Request Amount vs Total TAT",
        "Request Amount",
        "Total TAT (Hours)",
        "amount_vs_tat_scatter.png",
    )

    plot_boxplot_by_category(
        request_df,
        "Vendor_Type",
        "Total_TAT",
        "Total TAT by Vendor Type",
        "Vendor Type",
        "Hours",
        "tat_by_vendor_type_boxplot.png",
    )

    avg_tat_by_complexity = (
        request_df.groupby("Complexity_Score")["Total_TAT"]
        .mean()
        .sort_values(ascending=False)
    )
    plot_bar(
        avg_tat_by_complexity,
        "Average Total TAT by Complexity Score",
        "Complexity Score",
        "Hours",
        "avg_tat_by_complexity.png",
    )


def print_key_insights(stage_df: pd.DataFrame, request_df: pd.DataFrame) -> None:
    """Print a few direct headline insights."""
    print("\n=== KEY EDA INSIGHTS ===\n")

    top_stage = (
        stage_df.groupby("Stage")["Total_Stage_Time"]
        .mean()
        .sort_values(ascending=False)
        .head(1)
    )
    print("Highest average total stage time:")
    print(top_stage)

    top_bottleneck = request_df["Bottleneck_Stage"].value_counts().head(1)
    print("\nMost frequent bottleneck stage:")
    print(top_bottleneck)

    delayed_rate = request_df["Delayed_Flag"].mean() * 100
    print(f"\nDelayed request rate: {delayed_rate:.2f}%")

    top_type = (
        request_df.groupby("Request_Type")["Total_TAT"]
        .mean()
        .sort_values(ascending=False)
        .head(1)
    )
    print("\nHighest average TAT by request type:")
    print(top_type)

    top_dept = (
        request_df.groupby("Department_Requesting")["Total_TAT"]
        .mean()
        .sort_values(ascending=False)
        .head(1)
    )
    print("\nHighest average TAT by department:")
    print(top_dept)

    rework_rate = stage_df["Rework_Flag"].mean() * 100
    print(f"\nRework event rate: {rework_rate:.2f}%")


def main() -> None:
    """Run EDA summaries and export visuals."""
    stage_df, request_df = load_data()

    stage_summary, request_summary, bottleneck_summary = save_summary_tables(
        stage_df,
        request_df,
    )

    print("\nStage Delay Summary:\n", stage_summary)
    print("\nRequest Summary:\n", request_summary)
    print("\nBottleneck Summary:\n", bottleneck_summary.head(10))

    create_stage_level_visuals(stage_df, request_df)
    create_request_level_visuals(request_df)
    print_key_insights(stage_df, request_df)

    print("\nEDA completed.")
    print(f"Figures saved in: {FIGURES_DIR}")
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()