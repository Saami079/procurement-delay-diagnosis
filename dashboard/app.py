import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from html import escape
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

sys.path.append(str(SRC_DIR))

from utils import load_stage_data, load_request_data, load_model

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Procurement Decision Support System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown(
    """
<style>
    [data-testid="stAppViewContainer"], .stApp, .main {
        overflow: visible !important;
    }

    .block-container {
        /* FIXED: Increased padding-top to 100px to ensure the title is fully visible */
        padding-top: 100px; 
        padding-bottom: 1.2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1500px;
        overflow: visible !important;
    }

    html, body, [class*="css"] {
        font-family: "Segoe UI", Arial, sans-serif;
    }

    .main {
        background: linear-gradient(180deg, #0E1117 0%, #131A24 100%);
        color: #EAECEF;
    }

    [data-testid="stSidebar"] {
        background-color: #0B0F14;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-top: 0;
        margin-bottom: 0.35rem;
        line-height: 1.35;
        padding-top: 0.2rem;
        word-break: break-word;
        letter-spacing: -0.02em;
        overflow: visible !important;
    }

    .hero-subtitle {
        font-size: 1.28rem;
        font-weight: 700;
        color: #E8EDF2;
        margin-bottom: 1.35rem;
        line-height: 1.35;
    }

    .section-label {
        font-size: 0.88rem;
        font-weight: 700;
        color: #FFC300;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.15rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        color: #FFFFFF;
        margin-bottom: 0.9rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 18px 18px 12px 18px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
    }

    .metric-col {
        display: flex;
        flex-direction: column;
        height: 100%;
    }

    .metric-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #C8D0D8;
        margin-bottom: 0.35rem;
        min-height: 3rem;
        display: flex;
        align-items: flex-start;
        line-height: 1.35;
    }

    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(0, 212, 255, 0.35);
        border-radius: 16px;
        padding: 14px 14px 12px 14px;
        min-height: 120px;
        height: 100%;
        display: flex;
        align-items: stretch;
        justify-content: flex-start;
        width: 100%;
        box-sizing: border-box;
        flex: 1;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.25), 0 0 20px rgba(0, 212, 255, 0.10);
        transition: all 0.2s ease;
    }

    .metric-card-inner {
        width: 100%;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        gap: 0.45rem;
    }

    .metric-value {
        font-size: 1.02rem;
        font-weight: 600;
        color: #FFFFFF;
        line-height: 1.35;
        word-break: break-word;
        overflow-wrap: anywhere;
    }

    .metric-value-big {
        font-size: 1.45rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
        word-break: break-word;
        overflow-wrap: anywhere;
    }

    .metric-delta {
        margin-top: 0.35rem;
        color: #B8C1CC;
        font-size: 0.82rem;
    }

    .premium-insight {
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(255,195,0,0.30);
        border-left: 4px solid #FFC300;
        border-radius: 18px;
        padding: 18px 18px 12px 18px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 0 18px rgba(255,195,0,0.10);
        margin-top: 0.4rem;
        margin-bottom: 1rem;
    }

    .premium-insight-title {
        font-size: 1rem;
        font-weight: 700;
        color: #FFC300;
        margin-bottom: 0.55rem;
    }

    .premium-insight-text {
        font-size: 0.95rem;
        color: #E8EDF2;
        line-height: 1.55;
    }

    .prediction-box {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px;
        padding: 22px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.30);
    }

    .small-note {
        color: #A8B1BB;
        font-size: 0.86rem;
    }

    .micro-interpretation {
        font-size: 0.88rem;
        color: #B9C3CD;
        margin-top: 0.45rem;
        margin-bottom: 0.15rem;
        line-height: 1.5;
    }

    .recommendation-box {
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(34,197,94,0.25);
        border-left: 5px solid #22C55E;
        border-radius: 18px;
        padding: 18px 20px;
        margin-top: 0.9rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.22);
    }

    .recommendation-title {
        font-size: 1rem;
        font-weight: 700;
        color: #22C55E;
        margin-bottom: 0.45rem;
    }

    .recommendation-text {
        font-size: 0.98rem;
        color: #EAECEF;
        line-height: 1.55;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# COLORS
# ---------------------------------------------------
RED = "#FF6347"
TEAL = "#00D4FF"
BLUE = "#007BFF"
AMBER = "#FFC300"
GREEN = "#22C55E"
WHITE = "#FFFFFF"

TITLE_FONT = dict(size=20, family="Segoe UI, Arial, sans-serif", color=WHITE)


# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def clean_text(value):
    if value is None:
        return ""
    cleaned = str(value)
    for token in ["</div>", "<div>", "<div/>", "</div", "<div "]:
        cleaned = cleaned.replace(token, "")
    return escape(cleaned.strip())


def safe_html_metric(title: str, value: str, big: bool = False, delta: str = ""):
    value_class = "metric-value-big" if big else "metric-value"
    clean_title = clean_text(title)
    clean_value = clean_text(value)
    clean_delta = clean_text(delta)

    st.markdown(
        f'<div class="metric-header">{clean_title}</div>', unsafe_allow_html=True
    )
    if clean_delta:
        st.markdown(
            f'<div class="metric-card"><div class="metric-card-inner"><div class="{value_class}">{clean_value}</div><div class="metric-delta">{clean_delta}</div></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="metric-card"><div class="metric-card-inner"><div class="{value_class}">{clean_value}</div></div></div>',
            unsafe_allow_html=True,
        )


def clean_feature_names(feature_names):
    cleaned = (
        pd.Series(feature_names, dtype="string")
        .fillna("")
        .str.replace("cat__", "", regex=False)
        .str.replace("num__", "", regex=False)
        .str.replace("cat_", "", regex=False)
        .str.replace("num_", "", regex=False)
        .str.replace("_", " ", regex=False)
        .str.strip()
        .str.title()
    )

    cleaned = cleaned.replace(
        {
            "Request Amount": "Request Amount",
            "Is High Complexity": "High Complexity",
            "Is High Value Request": "High Value Request",
            "Num Stages": "Number Of Stages",
            "Total Waiting": "Total Waiting Time",
            "Total Processing": "Total Processing Time",
            "Max Stage Delay": "Max Stage Delay",
            "Sla Breach Hours": "SLA Breach Hours",
            "Delay Ratio": "Delay Ratio",
            "Priority Medium": "Priority Medium",
            "Priority High": "Priority High",
            "Priority Low": "Priority Low",
            "Request Type Office Supplies": "Office Supplies",
            "Request Type Vendor Contract": "Vendor Contract",
            "Request Type It Purchase": "IT Purchase",
            "Request Type Equipment": "Equipment",
            "Vendor Type External": "External Vendor",
            "Vendor Type Internal": "Internal Vendor",
            "Department Requesting Hr": "HR Department",
            "Department Requesting It": "IT Department",
        }
    )

    return cleaned


def get_top_recommendation(worst_stage, primary_delay_driver, filtered_request_df):
    if primary_delay_driver == "Processing Time":
        top_recommendation = f"Improve execution capacity at {worst_stage}"
    elif primary_delay_driver == "Waiting Time":
        top_recommendation = f"Optimize queue flow before {worst_stage}"
    else:
        top_recommendation = f"Investigate process inefficiencies at {worst_stage}"

    if worst_stage == "Finance Approval":
        top_recommendation += " — simplify policy or add parallel validation"

    external_share = (
        filtered_request_df["Vendor_Type"]
        .value_counts(normalize=True)
        .get("External", 0)
    )
    if external_share > 0.5:
        top_recommendation += " — strengthen external vendor coordination"

    return top_recommendation


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
stage_df = load_stage_data()
request_df = load_request_data()
model, feature_cols = load_model()

# ---------------------------------------------------
# DATA PREP
# ---------------------------------------------------
for col in ["Start_Time", "End_Time"]:
    if col in stage_df.columns:
        stage_df[col] = pd.to_datetime(stage_df[col], errors="coerce")

for col in ["Request_Start", "Request_End"]:
    if col in request_df.columns:
        request_df[col] = pd.to_datetime(request_df[col], errors="coerce")

date_col = None
if "Request_Start" in request_df.columns:
    date_col = "Request_Start"
elif "Request_End" in request_df.columns:
    date_col = "Request_End"

if "Delay_Ratio" not in request_df.columns and {"Total_TAT", "SLA_Hours"}.issubset(
    request_df.columns
):
    request_df["Delay_Ratio"] = np.where(
        request_df["SLA_Hours"] > 0,
        request_df["Total_TAT"] / request_df["SLA_Hours"],
        np.nan,
    )

if "SLA_Breach_Hours" not in request_df.columns and {"Total_TAT", "SLA_Hours"}.issubset(
    request_df.columns
):
    request_df["SLA_Breach_Hours"] = np.maximum(
        request_df["Total_TAT"] - request_df["SLA_Hours"], 0
    )

if (
    "Is_High_Value_Request" not in request_df.columns
    and "Request_Amount" in request_df.columns
):
    request_df["Is_High_Value_Request"] = (
        request_df["Request_Amount"] > 150000
    ).astype(int)

if (
    "Is_High_Complexity" not in request_df.columns
    and "Complexity_Score" in request_df.columns
):
    request_df["Is_High_Complexity"] = (request_df["Complexity_Score"] >= 4).astype(int)

if "Total_Stage_Time" not in stage_df.columns and {
    "Processing_Time",
    "Waiting_Time",
}.issubset(stage_df.columns):
    stage_df["Total_Stage_Time"] = (
        stage_df["Processing_Time"] + stage_df["Waiting_Time"]
    )

if "Rework_Flag" not in stage_df.columns and "Stage" in stage_df.columns:
    stage_df["Rework_Flag"] = (
        stage_df["Stage"]
        .astype(str)
        .str.contains("Rework", case=False, na=False)
        .astype(int)
    )

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.markdown("## 🎛 Global Filters")

department_options = ["All"] + sorted(
    request_df["Department_Requesting"].dropna().unique().tolist()
)
priority_options = ["All"] + sorted(request_df["Priority"].dropna().unique().tolist())
vendor_options = ["All"] + sorted(request_df["Vendor_Type"].dropna().unique().tolist())

selected_department = st.sidebar.selectbox("Department", department_options)
selected_priority = st.sidebar.selectbox("Priority", priority_options)
selected_vendor = st.sidebar.selectbox("Vendor Type", vendor_options)

if date_col:
    min_date = request_df[date_col].min().date()
    max_date = request_df[date_col].max().date()
    selected_dates = st.sidebar.date_input(
        "Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )
else:
    selected_dates = None

# ---------------------------------------------------
# FILTER LOGIC
# ---------------------------------------------------
filtered_request_df = request_df.copy()

if selected_department != "All":
    filtered_request_df = filtered_request_df[
        filtered_request_df["Department_Requesting"] == selected_department
    ]

if selected_priority != "All":
    filtered_request_df = filtered_request_df[
        filtered_request_df["Priority"] == selected_priority
    ]

if selected_vendor != "All":
    filtered_request_df = filtered_request_df[
        filtered_request_df["Vendor_Type"] == selected_vendor
    ]

if date_col and isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered_request_df = filtered_request_df[
        (filtered_request_df[date_col].dt.date >= start_date)
        & (filtered_request_df[date_col].dt.date <= end_date)
    ]

filtered_ids = filtered_request_df["Request_ID"].unique().tolist()
filtered_stage_df = stage_df[stage_df["Request_ID"].isin(filtered_ids)].copy()

if filtered_request_df.empty or filtered_stage_df.empty:
    st.error(
        "No data available for the current filter combination. Change the sidebar filters."
    )
    st.stop()

# ---------------------------------------------------
# DERIVED TABLES
# ---------------------------------------------------
stage_summary = (
    filtered_stage_df.groupby("Stage")[
        ["Processing_Time", "Waiting_Time", "Total_Stage_Time"]
    ]
    .mean()
    .sort_values("Total_Stage_Time", ascending=True)
    .reset_index()
)

bottleneck_stage_counts = (
    filtered_request_df["Bottleneck_Stage"].value_counts().reset_index()
)
bottleneck_stage_counts.columns = ["Bottleneck_Stage", "Count"]

department_tat = (
    filtered_request_df.groupby("Department_Requesting")["Total_TAT"]
    .mean()
    .sort_values(ascending=True)
    .reset_index()
)

vendor_tat = (
    filtered_request_df.groupby("Vendor_Type")["Total_TAT"]
    .mean()
    .sort_values(ascending=True)
    .reset_index()
)

complexity_tat = (
    filtered_request_df.groupby("Complexity_Score")["Total_TAT"]
    .mean()
    .sort_values(ascending=True)
    .reset_index()
)

rework_cases = (
    filtered_stage_df.groupby("Request_Type")["Rework_Flag"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

# ---------------------------------------------------
# KPI CALCULATIONS
# ---------------------------------------------------
total_requests = int(filtered_request_df["Request_ID"].nunique())
delayed_pct = float(filtered_request_df["Delayed_Flag"].mean() * 100)
avg_tat = float(filtered_request_df["Total_TAT"].mean())
avg_sla = float(filtered_request_df["SLA_Hours"].mean())
sla_delta = avg_tat - avg_sla

worst_stage_row = stage_summary.sort_values("Total_Stage_Time", ascending=False).iloc[0]
worst_stage = str(worst_stage_row["Stage"])
worst_processing = float(worst_stage_row["Processing_Time"])
worst_waiting = float(worst_stage_row["Waiting_Time"])
primary_delay_driver = (
    "Processing Time" if worst_processing >= worst_waiting else "Waiting Time"
)

top_recommendation = get_top_recommendation(
    worst_stage, primary_delay_driver, filtered_request_df
)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown(
    '<div class="hero-title">Procurement Decision Support System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">Bottleneck diagnosis, delay-risk prediction, and operational recommendations</div>',
    unsafe_allow_html=True,
)

# ===================================================
# SECTION 1: EXECUTIVE SUMMARY
# ===================================================
st.markdown('<div class="section-label">Section 1</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">Executive Summary</div>', unsafe_allow_html=True
)

k1, k2, k3, k4 = st.columns(4, gap="large")

with k1:
    safe_html_metric("📦 Total Requests", f"{total_requests}", big=True)
with k2:
    safe_html_metric("🚨 Delayed %", f"{delayed_pct:.2f}%", big=True)
with k3:
    safe_html_metric(
        "⏱ Avg TAT vs SLA", f"{avg_tat:.2f} hrs", delta=f"{sla_delta:.2f} hrs vs SLA"
    )
with k4:
    safe_html_metric("🔥 Current Worst Bottleneck", worst_stage)

st.markdown(
    f"""
<div class="recommendation-box">
    <div class="recommendation-title">✅ Top Recommendation</div>
    <div class="recommendation-text"><b>{escape(f"Improve execution capacity at {worst_stage}")}</b> — simplify policy or add parallel validation — strengthen external vendor coordination.</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="micro-interpretation">This section gives an executive snapshot of workflow health: workload volume, delay level, turnaround performance, the dominant bottleneck, and the immediate action priority.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# KILLER INSIGHT BLOCK
# ---------------------------------------------------
st.markdown("### ⚡ Key Insight")

top_delay_contributors = filtered_request_df.sort_values("Total_TAT", ascending=False)
top_20_pct = max(int(len(top_delay_contributors) * 0.2), 1)

high_impact_delay = top_delay_contributors.head(top_20_pct)["Total_TAT"].sum()
total_delay = filtered_request_df["Total_TAT"].sum()
impact_pct = (high_impact_delay / total_delay) * 100 if total_delay > 0 else 0

st.markdown(
    f"""
<div class="premium-insight">
    <div class="premium-insight-title">Delay Concentration Insight</div>
    <div class="premium-insight-text">
        Top 20% of requests contribute <b>{impact_pct:.2f}%</b> of total workflow delay.<br><br>
        This shows that inefficiency is concentrated in a relatively small group of high-friction requests,
        not evenly distributed across the workflow.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ===================================================
# SECTION 2: BOTTLENECK ANALYSIS
# ===================================================
st.markdown('<div class="section-label">Section 2</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">Bottleneck Analysis</div>', unsafe_allow_html=True
)

b1, b2 = st.columns((1.25, 1))

with b1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    processing_color_map = [
        RED if stage == "Finance Approval" else TEAL for stage in stage_summary["Stage"]
    ]
    waiting_color_map = [
        AMBER if stage == "Finance Approval" else BLUE
        for stage in stage_summary["Stage"]
    ]

    fig_stage = go.Figure()
    fig_stage.add_trace(
        go.Bar(
            y=stage_summary["Stage"],
            x=stage_summary["Processing_Time"],
            name="Processing Time",
            orientation="h",
            marker=dict(color=processing_color_map),
        )
    )
    fig_stage.add_trace(
        go.Bar(
            y=stage_summary["Stage"],
            x=stage_summary["Waiting_Time"],
            name="Waiting Time",
            orientation="h",
            marker=dict(color=waiting_color_map),
        )
    )

    fig_stage.update_layout(
        title=dict(text="Stage Time Composition", font=TITLE_FONT),
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family="Segoe UI, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", y=1.10),
        xaxis_title="Hours",
        yaxis_title="",
    )
    st.plotly_chart(fig_stage, use_container_width=True)
    st.markdown(
        '<div class="micro-interpretation">The stacked view separates actual processing time from waiting time, making it easier to distinguish execution bottlenecks from queue-related delay.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with b2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig_donut = px.pie(
        bottleneck_stage_counts,
        names="Bottleneck_Stage",
        values="Count",
        hole=0.58,
        color="Bottleneck_Stage",
        color_discrete_sequence=[RED, TEAL, AMBER, BLUE, GREEN, "#6C5CE7"],
    )
    fig_donut.update_layout(
        title=dict(text="Bottleneck Share", font=TITLE_FONT),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family="Segoe UI, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=55, b=10),
        showlegend=True,
    )
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown(
        '<div class="micro-interpretation">A larger share indicates the stage that absorbs the highest proportion of overall workflow delay.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
<div class="premium-insight">
    <div class="premium-insight-title">Dynamic Bottleneck Interpretation</div>
    <div class="premium-insight-text">
        <b>Worst Performing Stage:</b> {worst_stage}<br>
        <b>Primary Delay Driver:</b> {primary_delay_driver}<br>
        <b>Average Stage Time:</b> {worst_stage_row['Total_Stage_Time']:.2f} hours<br><br>
        Under the active filters, the workflow is most constrained at <b>{worst_stage}</b>.
        The dominant contributor is <b>{primary_delay_driver.lower()}</b>, which means the issue is being created inside the stage, not just inherited from upstream.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ===================================================
# SECTION 3: COMPLEXITY & DELAY DRIVERS
# ===================================================
st.markdown('<div class="section-label">Section 3</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">Complexity & Delay Drivers</div>',
    unsafe_allow_html=True,
)

c_row1, c_row2 = st.columns((1.25, 1))

with c_row1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    scatter_df = filtered_request_df.copy()
    scatter_df["Complexity_Label"] = scatter_df["Complexity_Score"].astype(str)

    fig_scatter = px.scatter(
        scatter_df,
        x="Request_Amount",
        y="Total_TAT",
        color="Complexity_Label",
        opacity=0.65,
        color_discrete_map={"1": TEAL, "2": BLUE, "3": "#5BC0EB", "4": AMBER, "5": RED},
        labels={
            "Request_Amount": "Request Amount",
            "Total_TAT": "Total TAT (Hours)",
            "Complexity_Label": "Complexity Score",
        },
    )
    fig_scatter.update_layout(
        title=dict(text="Request Amount vs Total TAT", font=TITLE_FONT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family="Segoe UI, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=55, b=10),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown(
        '<div class="micro-interpretation">Higher-value and higher-complexity requests show stronger delay tendency, suggesting approval burden rises with business criticality.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c_row2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig_complexity = px.bar(
        complexity_tat,
        x="Complexity_Score",
        y="Total_TAT",
        color="Total_TAT",
        color_continuous_scale=[TEAL, AMBER, RED],
    )
    fig_complexity.update_layout(
        title=dict(text="Complexity Distribution", font=TITLE_FONT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family="Segoe UI, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="Complexity Score",
        yaxis_title="Average TAT (Hours)",
    )
    st.plotly_chart(fig_complexity, use_container_width=True)
    st.markdown(
        '<div class="micro-interpretation">As complexity rises, average turnaround time increases, confirming that complexity is a structural delay driver.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

c_row3, c_row4 = st.columns(2)

with c_row3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig_vendor = px.bar(
        vendor_tat,
        y="Vendor_Type",
        x="Total_TAT",
        orientation="h",
        color="Vendor_Type",
        color_discrete_map={"Internal": TEAL, "External": AMBER},
    )
    fig_vendor.update_layout(
        title=dict(text="Vendor Comparison", font=TITLE_FONT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family="Segoe UI, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=55, b=10),
        yaxis_title="Vendor Type",
        xaxis_title="Average TAT (Hours)",
        showlegend=False,
    )
    st.plotly_chart(fig_vendor, use_container_width=True)
    st.markdown(
        '<div class="micro-interpretation">External vendor cases usually require extra validation and coordination, making them more delay-prone.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c_row4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig_dept = px.bar(
        department_tat,
        y="Department_Requesting",
        x="Total_TAT",
        orientation="h",
        color="Total_TAT",
        color_continuous_scale=[TEAL, AMBER, RED],
    )
    fig_dept.update_layout(
        title=dict(text="Department Comparison", font=TITLE_FONT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family="Segoe UI, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=55, b=10),
        yaxis_title="Department",
        xaxis_title="Average TAT (Hours)",
    )
    st.plotly_chart(fig_dept, use_container_width=True)
    st.markdown(
        '<div class="micro-interpretation">Departments with longer turnaround may need tighter prioritization rules, clearer request quality, or cleaner approval handoffs.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ===================================================
# SECTION 4: TIME-BASED TREND ANALYSIS
# ===================================================
st.markdown('<div class="section-label">Section 4</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">Time-Based Trend Analysis</div>', unsafe_allow_html=True
)

if date_col:
    trend_df = filtered_request_df.copy()
    trend_df["Date"] = trend_df[date_col].dt.date

    daily_tat_trend = trend_df.groupby("Date")["Total_TAT"].mean().reset_index()
    daily_delay_trend = trend_df.groupby("Date")["Delayed_Flag"].mean().reset_index()

    t1, t2 = st.columns(2)

    with t1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_tat_trend = px.line(daily_tat_trend, x="Date", y="Total_TAT", markers=True)
        fig_tat_trend.update_traces(
            line=dict(color="#1E90FF", width=3), marker=dict(color="#1E90FF", size=7)
        )
        fig_tat_trend.update_layout(
            title=dict(text="Average TAT Over Time", font=TITLE_FONT),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=WHITE, family="Segoe UI, Arial, sans-serif"),
            margin=dict(l=10, r=10, t=55, b=10),
            xaxis_title="Date",
            yaxis_title="Average TAT (Hours)",
        )
        st.plotly_chart(fig_tat_trend, use_container_width=True)
        st.markdown(
            '<div class="micro-interpretation">This trend reveals whether turnaround time is stable or spikes during specific operational periods.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_delay_trend = px.line(
            daily_delay_trend, x="Date", y="Delayed_Flag", markers=True
        )
        fig_delay_trend.update_traces(
            line=dict(color="#FF7F50", width=3), marker=dict(color="#FF7F50", size=7)
        )
        fig_delay_trend.update_layout(
            title=dict(text="Delay Rate Over Time", font=TITLE_FONT),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=WHITE, family="Segoe UI, Arial, sans-serif"),
            margin=dict(l=10, r=10, t=55, b=10),
            xaxis_title="Date",
            yaxis_title="Delay Rate",
        )
        st.plotly_chart(fig_delay_trend, use_container_width=True)
        st.markdown(
            '<div class="micro-interpretation">The delay-rate line helps identify persistent inefficiency versus isolated bursts of workflow disruption.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Time trend not available due to missing date fields.")

st.markdown("<br>", unsafe_allow_html=True)

# ===================================================
# SECTION 5: REQUEST TYPE BEHAVIOR
# ===================================================
st.markdown('<div class="section-label">Section 5</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">Request Type Behavior</div>', unsafe_allow_html=True
)

r1, r2 = st.columns((1.2, 1))

with r1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig_box = px.box(
        filtered_request_df,
        x="Request_Type",
        y="Total_TAT",
        color="Request_Type",
        points="all",
        color_discrete_sequence=[TEAL, BLUE, AMBER, RED],
    )
    fig_box.update_traces(jitter=0.35, pointpos=0, marker=dict(size=4, opacity=0.25))
    fig_box.update_layout(
        title=dict(text="TAT Spread by Request Type", font=TITLE_FONT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family="Segoe UI, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="Request Type",
        yaxis_title="Total TAT (Hours)",
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown(
        '<div class="micro-interpretation">This view shows both spread and outliers, revealing which request types are consistently slower and which ones are unstable.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with r2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if rework_cases["Rework_Flag"].sum() > 0:
        fig_rework = px.bar(
            rework_cases,
            y="Request_Type",
            x="Rework_Flag",
            orientation="h",
            color="Rework_Flag",
            color_continuous_scale=[TEAL, AMBER, RED],
        )
        fig_rework.update_layout(
            title=dict(text="Rework-Heavy Categories", font=TITLE_FONT),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=WHITE, family="Segoe UI, Arial, sans-serif"),
            margin=dict(l=10, r=10, t=55, b=10),
            yaxis_title="Request Type",
            xaxis_title="Rework Count",
        )
        st.plotly_chart(fig_rework, use_container_width=True)
        st.markdown(
            '<div class="micro-interpretation">Higher rework count signals repeated clarification loops, which increase turnaround variability and operational waste.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Rework category summary is not available for the current filter view.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ===================================================
# SECTION 6: PREDICTIVE RISK
# ===================================================
st.markdown('<div class="section-label">Section 6</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Predictive Risk</div>', unsafe_allow_html=True)

outer_left, outer_mid, outer_right = st.columns([1, 2, 1])

with outer_mid:
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.markdown("### 🧠 Delay Risk Predictor")
    st.markdown(
        '<div class="small-note">Enter request characteristics to estimate delay probability and understand operational risk before approval failure becomes visible.</div>',
        unsafe_allow_html=True,
    )

    f1, f2 = st.columns(2)

    with f1:
        priority = st.selectbox("Priority", ["Low", "Medium", "High"])
        request_type = st.selectbox(
            "Request Type",
            ["IT Purchase", "Office Supplies", "Vendor Contract", "Equipment"],
        )
        department = st.selectbox(
            "Department Requesting", ["IT", "HR", "Operations", "Finance", "Admin"]
        )
        vendor_type = st.selectbox("Vendor Type", ["Internal", "External"])

    with f2:
        request_amount = st.number_input(
            "Request Amount", min_value=1000, max_value=250000, value=50000, step=1000
        )
        complexity_score = st.slider("Complexity Score", 1, 5, 3)
        system_load = st.slider("System Load", 0.0, 1.0, 0.5, 0.1)
        num_stages = st.slider("Number of Stages", 3, 8, 4)

    s1, s2, s3 = st.columns(3)
    with s1:
        total_processing = st.number_input(
            "Estimated Total Processing Hours", min_value=1.0, value=120.0
        )
    with s2:
        total_waiting = st.number_input(
            "Estimated Total Waiting Hours", min_value=0.0, value=30.0
        )
    with s3:
        max_stage_delay = st.number_input(
            "Maximum Single Stage Delay", min_value=1.0, value=80.0
        )

    is_high_value = 1 if request_amount > 150000 else 0
    is_high_complexity = 1 if complexity_score >= 4 else 0

    estimated_sla = 48 + complexity_score * 24
    if priority == "High":
        estimated_sla *= 0.8
    elif priority == "Low":
        estimated_sla *= 1.2

    total_tat_est = total_processing + total_waiting
    delay_ratio = total_tat_est / estimated_sla if estimated_sla > 0 else 0
    sla_breach_hours = max(total_tat_est - estimated_sla, 0)

    input_df = pd.DataFrame(
        [
            {
                "Priority": priority,
                "Request_Type": request_type,
                "Department_Requesting": department,
                "Vendor_Type": vendor_type,
                "Request_Amount": request_amount,
                "Complexity_Score": complexity_score,
                "System_Load": system_load,
                "Num_Stages": num_stages,
                "Total_Processing": total_processing,
                "Total_Waiting": total_waiting,
                "Max_Stage_Delay": max_stage_delay,
                "Is_High_Value_Request": is_high_value,
                "Is_High_Complexity": is_high_complexity,
                "Delay_Ratio": delay_ratio,
                "SLA_Breach_Hours": sla_breach_hours,
            }
        ]
    )

    if st.button("Predict Delay Risk", use_container_width=True):
        try:
            prediction = model.predict(input_df)[0]
            probability = float(model.predict_proba(input_df)[0][1])

            g1, g2 = st.columns([1.2, 1])

            with g1:
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        title={"text": "Delay Risk Probability"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": RED if probability >= 0.5 else TEAL},
                            "steps": [
                                {"range": [0, 40], "color": "rgba(0, 212, 255, 0.35)"},
                                {"range": [40, 70], "color": "rgba(255,195,0,0.35)"},
                                {"range": [70, 100], "color": "rgba(255,99,71,0.45)"},
                            ],
                        },
                    )
                )
                gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=WHITE, family="Segoe UI, Arial, sans-serif"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=320,
                )
                st.plotly_chart(gauge, use_container_width=True)

            with g2:
                if int(prediction) == 1:
                    st.error("⚠️ Risk Label: High Delay Risk")
                else:
                    st.success("✅ Risk Label: Lower Delay Risk")

                st.metric("Predicted Delay Probability", f"{probability * 100:.2f}%")
                st.metric("Estimated SLA", f"{estimated_sla:.2f} hrs")
                st.metric("Estimated SLA Breach", f"{sla_breach_hours:.2f} hrs")

                factor_lines = []
                if complexity_score >= 4:
                    factor_lines.append("High complexity increases approval burden.")
                if request_amount > 150000:
                    factor_lines.append("High request amount raises scrutiny level.")
                if total_waiting > total_processing * 0.4:
                    factor_lines.append(
                        "Waiting is significantly contributing to delay risk."
                    )
                if vendor_type == "External":
                    factor_lines.append(
                        "External vendor handling adds coordination overhead."
                    )
                if not factor_lines:
                    factor_lines.append(
                        "Current inputs indicate relatively moderate operational risk."
                    )

                st.markdown("**Factor Explanation**")
                for line in factor_lines:
                    st.markdown(f"- {line}")

        except Exception as e:
            st.error(
                f"Prediction failed. Check that your model pipeline matches the input schema. Details: {e}"
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🔬 Model Feature Importance")

try:
    rf_model = model.named_steps["classifier"]
    importances = rf_model.feature_importances_
    raw_feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    cleaned_feature_names = clean_feature_names(raw_feature_names)

    fi_df = (
        pd.DataFrame({"Feature": cleaned_feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(15)
    )

    fig_fi = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=[TEAL, AMBER, RED],
    )
    fig_fi.update_layout(
        title=dict(text="Top Features Influencing Delay Prediction", font=TITLE_FONT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family="Segoe UI, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=55, b=10),
        yaxis_title="Feature",
        xaxis_title="Importance",
    )
    fig_fi.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig_fi, use_container_width=True)

    top_features = fi_df.head(3)
    if len(top_features) >= 3:
        st.markdown(
            f"""
        <div class="micro-interpretation">
        Top 3 features are dominated by <b>{top_features.iloc[0]['Feature']}</b>,
        <b>{top_features.iloc[1]['Feature']}</b>, and
        <b>{top_features.iloc[2]['Feature']}</b>.
        This shows that delay risk is mainly driven by workload intensity, SLA pressure,
        and structural workflow conditions rather than isolated random events.
        </div>
        """,
            unsafe_allow_html=True,
        )

except Exception as e:
    st.info(f"Feature importance could not be generated. {e}")

st.markdown("<br>", unsafe_allow_html=True)

# ===================================================
# SECTION 7: RECOMMENDATIONS
# ===================================================
st.markdown('<div class="section-label">Section 7</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)

top_bottleneck = filtered_request_df["Bottleneck_Stage"].value_counts().idxmax()
top_request_type = (
    filtered_request_df.groupby("Request_Type")["Total_TAT"]
    .mean()
    .sort_values(ascending=False)
    .idxmax()
)
top_department = (
    filtered_request_df.groupby("Department_Requesting")["Total_TAT"]
    .mean()
    .sort_values(ascending=False)
    .idxmax()
)

rec1, rec2, rec3 = st.columns(3)

with rec1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Operational Actions")
    st.markdown(
        f"""
- Prioritize intervention at **{top_bottleneck}**
- Set escalation thresholds for requests approaching SLA breach
- Monitor high-risk requests earlier instead of reacting after delay becomes visible
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with rec2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Staffing / Process Suggestions")
    if primary_delay_driver == "Processing Time":
        st.markdown(
            f"""
- Review execution workload and review capacity at **{top_bottleneck}**
- Reassess staffing, batching, or approval capacity in **{top_department}**
- Reduce manual review burden for repetitive low-risk cases
        """
        )
    else:
        st.markdown(
            f"""
- Review queue buildup before **{top_bottleneck}**
- Tighten handoff rules and prioritization logic in **{top_department}**
- Reduce waiting caused by unclear ownership and stage transitions
        """
        )
    st.markdown("</div>", unsafe_allow_html=True)

with rec3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Stage-Specific Intervention Ideas")
    st.markdown(
        f"""
- Introduce pre-validation for **{top_request_type}** requests
- Simplify handoffs before **{top_bottleneck}**
- Add workflow rules to minimize unnecessary rework loops
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="micro-interpretation">Recommendations are dynamically aligned with the current filter context, making the dashboard useful for both executive review and operational diagnosis.</div>',
    unsafe_allow_html=True,
)

# ===================================================
# SECTION 8: DETAILED DATA VIEW
# ===================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Section 8</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">Detailed Data View</div>', unsafe_allow_html=True
)

d1, d2 = st.columns(2)

with d1:
    avg_tat_dept = (
        filtered_request_df.groupby("Department_Requesting")["Total_TAT"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )

    fig_dept_detail = px.bar(
        avg_tat_dept,
        y="Department_Requesting",
        x="Total_TAT",
        orientation="h",
        color="Total_TAT",
        color_continuous_scale=[TEAL, AMBER, RED],
    )
    fig_dept_detail.update_layout(
        title=dict(text="Average TAT by Department", font=TITLE_FONT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family="Segoe UI, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=55, b=10),
        yaxis_title="Department",
        xaxis_title="Average TAT (Hours)",
    )
    st.plotly_chart(fig_dept_detail, use_container_width=True)

with d2:
    avg_tat_vendor = (
        filtered_request_df.groupby("Vendor_Type")["Total_TAT"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )

    fig_vendor_detail = px.bar(
        avg_tat_vendor,
        y="Vendor_Type",
        x="Total_TAT",
        orientation="h",
        color="Vendor_Type",
        color_discrete_map={"Internal": TEAL, "External": AMBER},
    )
    fig_vendor_detail.update_layout(
        title=dict(text="Average TAT by Vendor Type", font=TITLE_FONT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family="Segoe UI, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=55, b=10),
        yaxis_title="Vendor Type",
        xaxis_title="Average TAT (Hours)",
        showlegend=False,
    )
    st.plotly_chart(fig_vendor_detail, use_container_width=True)

st.markdown("#### Filtered Request-Level Data")
st.dataframe(filtered_request_df, use_container_width=True, height=320)

# ---------------------------------------------------
# DOWNLOAD BUTTONS
# ---------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### ⬇ Export Filtered Data")

dl1, dl2 = st.columns(2)

with dl1:
    stage_csv = filtered_stage_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Stage-Level CSV",
        data=stage_csv,
        file_name="filtered_stage_level_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

with dl2:
    request_csv = filtered_request_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Request-Level CSV",
        data=request_csv,
        file_name="filtered_request_level_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
