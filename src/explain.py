"""
explain.py
----------
Generates SHAP (SHapley Additive exPlanations) feature importance plots.

SHAP answers the question: "Which clinical features pushed this patient's
predicted diabetes risk up or down, and by how much?"

This directly implements the Explainable AI (XAI) principle discussed in
the research report — making AI predictions interpretable to clinicians.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

os.makedirs("results", exist_ok=True)


def plot_shap_importance(model, X_test, feature_names, model_name="Best Model"):
    """
    Creates a SHAP bar chart showing mean absolute feature importance.

    Args:
        model        : Trained sklearn model (must support predict_proba)
        X_test       : Scaled test feature array (numpy array)
        feature_names: List of feature name strings
        model_name   : String label used in the plot title
    """

    print(f"[INFO] Computing SHAP values for: {model_name}...")

    # TreeExplainer is fast and exact for tree-based models (RF, GBM, DT)
    # For Logistic Regression we fall back to the slower but general Explainer
    try:
        explainer    = shap.TreeExplainer(model)
        shap_values  = explainer.shap_values(X_test)

        # For binary classification, shap_values is a list [class_0, class_1]
        # We want class 1 (diabetic)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

    except Exception:
        # Fallback for linear models (Logistic Regression)
        print("[INFO] Using KernelExplainer (slower — model is not tree-based)...")
        # Use a small background sample to keep it fast
        background   = shap.sample(X_test, 50, random_state=42)
        explainer    = shap.KernelExplainer(model.predict_proba, background)
        shap_values  = explainer.shap_values(X_test[:100])  # sample for speed
        shap_vals    = shap_values[1]

    # Mean absolute SHAP value per feature = overall importance
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    # Sort features from most to least important
    sorted_idx     = np.argsort(mean_abs_shap)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_values   = mean_abs_shap[sorted_idx]

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(sorted_features)))
    bars   = ax.barh(sorted_features, sorted_values, color=colors)

    # Value labels
    for bar, val in zip(bars, sorted_values):
        ax.text(
            val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9.5, color="#1E3A5F"
        )

    ax.set_xlabel("Mean |SHAP Value| — Average Impact on Prediction", fontsize=11)
    ax.set_title(
        f"SHAP Feature Importance — {model_name}\n"
        f"(Higher value = stronger influence on diabetes prediction)",
        fontsize=12, fontweight="bold", color="#0A1628"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[INFO] Saved → results/feature_importance.png")


def explain_single_patient(model, scaler, feature_names, patient_values):
    """
    Returns SHAP values for a single patient's prediction.
    Used by the Streamlit app to show per-patient explanation.

    Args:
        model         : Trained model
        scaler        : Fitted StandardScaler
        feature_names : List of feature name strings
        patient_values: Dict of {feature_name: value}

    Returns:
        shap_dict: {feature_name: shap_value} sorted by abs(shap_value)
    """
    # Build input array in the correct feature order
    input_array = np.array([[patient_values[f] for f in feature_names]])

    # Scale the input (same transformation as training)
    input_scaled = scaler.transform(input_array)

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]
    except Exception:
        background  = np.zeros((1, len(feature_names)))
        explainer   = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(input_scaled)
        sv = shap_values[1][0]

    # Return as sorted dict
    shap_dict = dict(zip(feature_names, sv))
    shap_dict = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True))
    return shap_dict
