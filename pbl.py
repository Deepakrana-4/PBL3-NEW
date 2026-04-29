#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         URBAN WATER DEMAND PREDICTION SYSTEM                     ║
║    Graphic Era University — PBL Project Prediction Script        ║
╚══════════════════════════════════════════════════════════════════╝

Mode 1: Input household features  ->  Predict daily water usage (Liters/day)
Mode 2: Explore dataset insights by Building Type or Income Level

Usage:
    python predict.py
"""

import pickle
import numpy as np
import pandas as pd
import os
import sys

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\dr488\OneDrive - Graphic Era University\Desktop\pbl\pbl\model.pkl"
DATA_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urban_water_demand.csv")

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║         URBAN WATER DEMAND PREDICTION SYSTEM                     ║
║    Graphic Era University  |  PBL Project  |  Linear Regression  ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────
def load_model():
    """Load the saved model pickle (model + scaler + feature_names)."""
    if not os.path.exists(MODEL_PATH):
        print(f"\n  ERROR: model.pkl not found at:\n     {MODEL_PATH}")
        print("  Ensure the path is correct and the notebook has been executed.\n")
        sys.exit(1)
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    return data


def load_dataset():
    """Load the raw CSV dataset for insight queries."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    alt = os.path.join(os.path.dirname(MODEL_PATH), "urban_water_demand.csv")
    if os.path.exists(alt):
        return pd.read_csv(alt)
    return None


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def separator(char='─', width=66):
    print(char * width)


def print_header(title):
    separator('═')
    print(f"  {title}")
    separator('═')


def get_float(prompt, lo=None, hi=None, default=None):
    """Prompt user for a float with optional range validation."""
    while True:
        raw = input(f"  {prompt}").strip()
        if raw == '' and default is not None:
            return float(default)
        try:
            val = float(raw)
            if lo is not None and val < lo:
                print(f"    WARNING: Value must be >= {lo}. Try again.")
                continue
            if hi is not None and val > hi:
                print(f"    WARNING: Value must be <= {hi}. Try again.")
                continue
            return val
        except ValueError:
            print("    WARNING: Please enter a valid number.")


def choose_from_list(prompt, options):
    """Present a numbered list and return the chosen item."""
    print(f"\n  {prompt}")
    for i, opt in enumerate(options, 1):
        print(f"    [{i}] {opt}")
    while True:
        raw = input("  Your choice (number): ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("    WARNING: Invalid choice. Try again.")


def categorize_usage(liters):
    """Classify predicted water usage into Low / Medium / High."""
    if liters < 500:
        return "LOW    (< 500 L/day)"
    elif liters < 1000:
        return "MEDIUM (500 - 1000 L/day)"
    else:
        return "HIGH   (> 1000 L/day)"


def usage_recommendation(liters):
    """Return a conservation recommendation based on predicted usage."""
    if liters < 500:
        return "Household is within an efficient water usage range. No immediate action required."
    elif liters < 1000:
        return ("Moderate usage detected. Consider reducing garden irrigation "
                "frequency and optimising appliance usage cycles.")
    else:
        return ("High usage detected. Inspect for leaks, upgrade to water-efficient "
                "appliances, and limit garden irrigation during peak temperature months.")


# ─────────────────────────────────────────────────────────────────
# MODE 1 — Predict Daily Water Usage
# ─────────────────────────────────────────────────────────────────
def mode_predict(artifacts):
    print_header("MODE 1  —  Predict Daily Water Usage (Liters/day)")
    print("\n  Enter the household details below.")
    print("  Press [Enter] to accept the default value shown in brackets.\n")
    separator()

    household_size = get_float("Household Size  (no. of people)         [default=4]   : ", 1,   30,   4)
    seasonal_index = get_float("Seasonal Index  (0.5 = dry, 1.5 = wet)  [default=1.0] : ", 0.1, 2.0, 1.0)
    garden_area    = get_float("Garden Area     (sq. metres, 0 = none)   [default=80]  : ", 0, 5000,  80)

    # Build feature vector — same order as training
    X_raw  = np.array([[household_size, seasonal_index, garden_area]], dtype=float)
    model  = artifacts['model']
    scaler = artifacts['scaler']

    try:
        X_scaled = scaler.transform(X_raw)
    except Exception:
        # Fallback: scaler may expect 7 features due to notebook inconsistency
        padded = np.zeros((1, 7))
        padded[0, :3] = X_raw[0]
        X_scaled = scaler.transform(padded)[:, :3]

    prediction = max(0, round(model.predict(X_scaled)[0], 1))
    category   = categorize_usage(prediction)
    reco       = usage_recommendation(prediction)

    separator()
    print(f"""
  ╔═══════════════════════════════════════════════════════════╗
  ║  PREDICTED DAILY WATER USAGE                             ║
  ║                                                           ║
  ║      Predicted Value  :  {prediction:>8.1f} Liters / day         ║
  ║      Usage Category   :  {category:<34} ║
  ╚═══════════════════════════════════════════════════════════╝

  Inputs Provided:
       Household Size  : {household_size} people
       Seasonal Index  : {seasonal_index}
       Garden Area     : {garden_area} sq. metres

  Recommendation:
       {reco}
    """)

    monthly = prediction * 30
    yearly  = prediction * 365
    print(f"  Estimated Monthly Usage  :  {monthly:>10,.1f} Liters")
    print(f"  Estimated Yearly Usage   :  {yearly:>10,.1f} Liters\n")


# ─────────────────────────────────────────────────────────────────
# MODE 2 — Dataset Insights
# ─────────────────────────────────────────────────────────────────
def mode_insights():
    print_header("MODE 2  —  Dataset Insights & Condition Analysis")

    df = load_dataset()
    if df is None:
        print(f"\n  ERROR: Could not load urban_water_demand.csv.")
        print(f"  Checked location: {DATA_PATH}")
        print("  Place the CSV file next to this script or update DATA_PATH.\n")
        return

    filter_choice = choose_from_list(
        "Select how you would like to explore the data:",
        ["By Building Type", "By Income Level", "Overall Summary"]
    )

    separator()

    if filter_choice == "By Building Type":
        options  = sorted(df['Building_Type'].dropna().unique())
        selected = choose_from_list("Select a Building Type:", options)
        subset   = df[df['Building_Type'] == selected].copy()
        label    = f"BUILDING TYPE : {selected.upper()}"

    elif filter_choice == "By Income Level":
        options  = sorted(df['Income_Level'].dropna().unique())
        selected = choose_from_list("Select an Income Level:", options)
        subset   = df[df['Income_Level'] == selected].copy()
        label    = f"INCOME LEVEL  : {selected.upper()}"

    else:
        subset = df.copy()
        label  = "OVERALL DATASET SUMMARY"

    # Statistics table
    separator()
    print(f"\n  INSIGHTS  —  {label}")
    print(f"  Total Records : {len(subset)}\n")

    stat_cols = {
        'Daily_Liters_Used' : 'Daily Water Used (L/day)',
        'Household_Size'    : 'Household Size (people)',
        'Seasonal_Index'    : 'Seasonal Index',
        'Garden_Area'       : 'Garden Area (sq.m)',
        'Avg_Temperature_C' : 'Avg Temperature (C)',
        'Num_Appliances'    : 'Number of Appliances',
    }

    print(f"  {'Feature':<30} {'Mean':>9} {'Median':>9} {'Std Dev':>9} {'Min':>8} {'Max':>8}")
    separator()
    for col, col_label in stat_cols.items():
        if col in subset.columns:
            s = subset[col].dropna()
            if len(s) > 0:
                print(f"  {col_label:<30} {s.mean():>9.2f} {s.median():>9.2f} "
                      f"{s.std():>9.2f} {s.min():>8.2f} {s.max():>8.2f}")

    separator()

    # Usage category breakdown
    if 'Daily_Liters_Used' in subset.columns:
        s      = subset['Daily_Liters_Used'].dropna()
        total  = len(s)
        low    = (s < 500).sum()
        medium = ((s >= 500) & (s < 1000)).sum()
        high   = (s >= 1000).sum()

        print(f"\n  Water Usage Category Breakdown:")
        print(f"    Low    (< 500 L/day)     : {low:>4} households  ({low/total*100:.1f}%)")
        print(f"    Medium (500-1000 L/day)  : {medium:>4} households  ({medium/total*100:.1f}%)")
        print(f"    High   (> 1000 L/day)    : {high:>4} households  ({high/total*100:.1f}%)")

    # Averages by group (overall mode only)
    if filter_choice == "Overall Summary":
        separator()
        if 'Building_Type' in df.columns:
            bt_avg = df.groupby('Building_Type')['Daily_Liters_Used'].mean().sort_values(ascending=False)
            print(f"\n  Average Daily Water Usage by Building Type:")
            for bt, avg in bt_avg.items():
                bar = '|' * int(avg / 50)
                print(f"    {bt:<14}: {avg:>7.1f} L/day  {bar}")

        if 'Income_Level' in df.columns:
            il_avg = df.groupby('Income_Level')['Daily_Liters_Used'].mean().sort_values(ascending=False)
            print(f"\n  Average Daily Water Usage by Income Level:")
            for il, avg in il_avg.items():
                bar = '|' * int(avg / 50)
                print(f"    {il:<14}: {avg:>7.1f} L/day  {bar}")

    # Median (ideal) conditions
    separator()
    print(f"\n  MEDIAN CONDITIONS FOR SELECTED GROUP:")
    for col, col_label in stat_cols.items():
        if col in subset.columns and col != 'Daily_Liters_Used':
            print(f"    {col_label:<30}: {subset[col].median():.2f}")
    print()


# ─────────────────────────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────────────────────────
def main():
    print(BANNER)

    print("  Loading model artifacts ...", end=' ', flush=True)
    artifacts = load_model()
    print("Done.")
    print(f"  Model Type  : {type(artifacts['model']).__name__}")
    print(f"  Features    : {artifacts.get('feature_names', ['Household_Size', 'Seasonal_Index', 'Garden_Area'])}\n")

    while True:
        separator('═')
        print("\n  MAIN MENU")
        separator()
        print("  [1]  Predict daily water usage from household features")
        print("  [2]  Explore dataset insights (Building Type / Income Level)")
        print("  [3]  Exit")
        separator()
        choice = input("  Select option: ").strip()

        if choice == '1':
            mode_predict(artifacts)
        elif choice == '2':
            mode_insights()
        elif choice == '3':
            print("\n  Exiting Urban Water Demand Prediction System. Goodbye.\n")
            sys.exit(0)
        else:
            print("  WARNING: Invalid option. Please enter 1, 2, or 3.\n")

        input("\n  Press [Enter] to return to the main menu ...")


if __name__ == '__main__':
    main()
    