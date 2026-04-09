# Methodology

## Project Objective
The goal of this project is to diagnose bottlenecks and delays in procurement approval workflows using data-driven analysis and predictive modeling.

The project focuses on identifying where delays occur, why they occur, and how they can be predicted.

---

## Data Source

Due to the lack of access to real enterprise procurement data, a synthetic dataset was generated.

The dataset simulates:
- sequential approval workflows
- stage-level processing and waiting times
- rework loops
- SLA-based delay classification
- varying request characteristics (type, priority, vendor, amount)

---

## Data Structure

Two datasets were created:

### 1. Stage-Level Dataset
Each row represents a workflow stage.

### 2. Request-Level Dataset
Each row represents a complete request.

This structure enables both:
- micro-level analysis (stage delays)
- macro-level analysis (overall request behavior)

---

## Data Processing

The pipeline consists of:

1. Data Generation (`generate_data.py`)
2. Data Cleaning and Feature Engineering (`processing.py`)
3. Final Dataset Preparation (`finalize_data.py`)

Derived features include:
- Total turnaround time (Total_TAT)
- Delay ratio
- SLA breach hours
- Complexity score
- High-value and high-complexity flags
- Bottleneck stage identification

---

## Exploratory Data Analysis

EDA was conducted to understand:
- data distributions
- request characteristics
- variation across categories
- delay prevalence

---

## Bottleneck Analysis

Bottleneck analysis focused on:
- stage-wise delay contribution
- bottleneck frequency
- waiting vs processing dominance
- department and vendor effects
- rework impact
- delay concentration

---

## Predictive Modeling

A Random Forest classifier was used to predict whether a request would be delayed.

Important considerations:
- Target: Delayed_Flag
- Leakage features removed:
  - Delay_Ratio
  - SLA_Breach_Hours

Model evaluation included:
- confusion matrix
- precision
- recall
- F1-score

---

## Dashboard

A Streamlit dashboard was developed to:
- visualize bottlenecks
- analyze delay drivers
- explore request characteristics
- predict delay risk

---

## Summary

The methodology combines:
- synthetic data engineering
- exploratory analysis
- bottleneck diagnostics
- predictive modeling
- dashboard-based interpretation

to create a complete analytical workflow.