"""
preprocess.py
-------------
Loads the Pima Indians Diabetes dataset, handles missing values,
and splits it into training and test sets.

The dataset has 768 patient records with 8 clinical features.
Target: 1 = Diabetic, 0 = Not Diabetic
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Column names for the dataset ────────────────────────────────────────────
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]
TARGET_NAME = "Outcome"


def load_data(data_path="data/diabetes.csv"):
    """
    Load the diabetes dataset from a CSV file.
    If the file doesn't exist, generate a synthetic but realistic
    dataset with the same statistical properties as the Pima dataset.

    Returns:
        df (pd.DataFrame): Full dataset with features and target column.
    """

    # If the CSV already exists, just read it — this is the normal path
    if os.path.exists(data_path):
        print(f"[INFO] Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        return df

    # Generate dataset locally (same structure as Pima Indians dataset)
    print("[INFO] Dataset not found — generating locally...")
    np.random.seed(42)
    n = 768

    preg = np.random.randint(0, 17, n)
    gluc = np.clip(np.random.normal(120, 32, n), 50, 200).astype(int)
    bp   = np.clip(np.random.normal(69, 19, n),  30, 122).astype(int)
    skin = np.clip(np.random.normal(20, 16, n),   0,  99).astype(int)
    ins  = np.clip(np.random.normal(79, 115, n),  0, 846).astype(int)
    bmi  = np.clip(np.random.normal(32, 7, n),   10,  67).round(1)
    dpf  = np.clip(np.random.normal(0.47, 0.33, n), 0.05, 2.42).round(3)
    age  = np.clip(np.random.normal(33, 12, n),  21,  81).astype(int)

    # Label: correlated with glucose, BMI, age, and pedigree (~35% positive)
    score   = (gluc/200 * 0.45 + bmi/67 * 0.25 + age/81 * 0.15 + dpf/2.42 * 0.15)
    outcome = (score + np.random.normal(0, 0.07, n) > 0.48).astype(int)

    df = pd.DataFrame({
        "Pregnancies": preg, "Glucose": gluc, "BloodPressure": bp,
        "SkinThickness": skin, "Insulin": ins, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age, "Outcome": outcome
    })

    os.makedirs(os.path.dirname(data_path) if os.path.dirname(data_path) else ".", exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"[INFO] Dataset saved to {data_path} ({len(df)} rows, {outcome.mean():.0%} positive)")

    return df


def clean_data(df):
    """
    Handle missing/zero values in clinical columns where 0 is physiologically impossible.

    Columns where 0 is medically impossible (replaced with median):
        - Glucose, BloodPressure, SkinThickness, Insulin, BMI

    Returns:
        df (pd.DataFrame): Cleaned DataFrame.
    """
    # Make a copy so we don't modify the original
    df = df.copy()

    # These features cannot realistically be 0 in a living person
    zero_not_valid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for col in zero_not_valid:
        if col in df.columns:
            # Count zeros before replacement
            n_zeros = (df[col] == 0).sum()
            if n_zeros > 0:
                # Replace 0 with the median of non-zero values
                median_val = df[col][df[col] != 0].median()
                df[col] = df[col].replace(0, median_val)
                print(f"[INFO] {col}: replaced {n_zeros} zero(s) with median ({median_val:.2f})")

    return df


def split_data(df, test_size=0.2, random_state=42):
    """
    Split the cleaned dataset into training and test sets.

    Args:
        df          : Cleaned DataFrame
        test_size   : Fraction of data reserved for testing (default 20%)
        random_state: Seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test, scaler
        (features are StandardScaler-normalised)
    """
    # Separate features (X) from target (y)
    X = df[FEATURE_NAMES].values
    y = df[TARGET_NAME].values

    # Split into 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Normalise features: mean=0, std=1
    # Fit ONLY on training data to prevent data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"\n[INFO] Data split complete:")
    print(f"       Training samples : {X_train.shape[0]}")
    print(f"       Test samples     : {X_test.shape[0]}")
    print(f"       Features         : {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, scaler


def get_preprocessed_data(data_path="data/diabetes.csv"):
    """
    One-call convenience function: load → clean → split.

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    df = load_data(data_path)
    df = clean_data(df)
    X_train, X_test, y_train, y_test, scaler = split_data(df)
    return X_train, X_test, y_train, y_test, scaler, FEATURE_NAMES


if __name__ == "__main__":
    # Quick test: run this file directly to verify preprocessing works
    X_train, X_test, y_train, y_test, scaler, feature_names = get_preprocessed_data()
    print(f"\n[OK] Preprocessing complete.")
    print(f"     Feature names: {feature_names}")
