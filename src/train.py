"""
train.py
--------
Trains four machine learning models on the diabetes dataset,
evaluates each one, prints a comparison table, saves plots
to results/, and saves the best model to models/best_model.pkl.

Run from the project root:
    python src/train.py
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no screen needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# Add project root to path so we can import src/preprocess.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import get_preprocessed_data

warnings.filterwarnings("ignore")

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
os.makedirs("models",  exist_ok=True)

# ── Colour palette matching the project report ────────────────────────────────
COLORS = ["#0EA5E9", "#059669", "#D97706", "#DC2626"]


def define_models():
    """
    Returns a dictionary of model name → (untrained) model object.
    All models use random_state=42 for reproducibility.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,    # enough iterations to converge
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,      # limit depth to avoid overfitting
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, # 100 decision trees in the ensemble
            max_depth=10,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ),
    }


def evaluate_model(model, X_test, y_test):
    """
    Compute standard classification metrics for a trained model.

    Returns a dict with: accuracy, precision, recall, f1, roc_auc
    """
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1

    return {
        "Accuracy" : round(accuracy_score (y_test, y_pred)                * 100, 2),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0)* 100, 2),
        "Recall"   : round(recall_score   (y_test, y_pred)                * 100, 2),
        "F1 Score" : round(f1_score       (y_test, y_pred)                * 100, 2),
        "ROC AUC"  : round(roc_auc_score  (y_test, y_pred_prob)           * 100, 2),
    }


def plot_model_comparison(results_df):
    """
    Bar chart comparing all four models across all five metrics.
    Saved to results/model_comparison.png
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    x       = np.arange(len(metrics))
    width   = 0.18

    for i, (model_name, row) in enumerate(results_df.iterrows()):
        bars = ax.bar(
            x + i * width,
            [row[m] for m in metrics],
            width,
            label=model_name,
            color=COLORS[i],
            alpha=0.88
        )
        # Add value labels on top of each bar
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=7.5, color="#1E3A5F"
            )

    ax.set_xlabel("Metric", fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Model Performance Comparison — Diabetes Prediction", fontsize=13, fontweight="bold", color="#0A1628")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.6)

    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[INFO] Saved → results/model_comparison.png")


def plot_roc_curves(models_trained, X_test, y_test):
    """
    ROC curve for all four models on the same axes.
    Saved to results/roc_curve.png
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for (name, model), color in zip(models_trained.items(), COLORS):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=color, linewidth=2)

    # Random classifier baseline (diagonal line)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold", color="#0A1628")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E2E8F0", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("results/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[INFO] Saved → results/roc_curve.png")


def plot_confusion_matrix(best_model, best_name, X_test, y_test):
    """
    Confusion matrix heatmap for the best model.
    Saved to results/confusion_matrix.png
    """
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"],
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=12, fontweight="bold", color="#0A1628")

    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[INFO] Saved → results/confusion_matrix.png")


def main():
    print("=" * 60)
    print("  AI Healthcare — Diabetes Prediction Pipeline")
    print("=" * 60)

    # ── Step 1: Load and preprocess data ───────────────────────────
    print("\n[STEP 1] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = get_preprocessed_data()

    # ── Step 2: Train all models ────────────────────────────────────
    print("\n[STEP 2] Training models...")
    models = define_models()
    models_trained = {}
    results = {}

    for name, model in models.items():
        print(f"         Training: {name}...", end=" ")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        models_trained[name] = model
        results[name] = metrics
        print(f"Accuracy: {metrics['Accuracy']}%  |  AUC: {metrics['ROC AUC']}%")

    # ── Step 3: Results table ───────────────────────────────────────
    print("\n[STEP 3] Model Performance Summary:")
    print("-" * 65)
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())
    print("-" * 65)

    # ── Step 4: Identify best model (by AUC) ───────────────────────
    best_name  = results_df["ROC AUC"].idxmax()
    best_model = models_trained[best_name]
    print(f"\n[STEP 4] Best model: {best_name}  (AUC: {results_df.loc[best_name,'ROC AUC']}%)")

    # ── Step 5: Save best model + scaler ───────────────────────────
    print("\n[STEP 5] Saving best model...")
    joblib.dump(
        {"model": best_model, "scaler": scaler, "features": feature_names, "name": best_name},
        "models/best_model.pkl"
    )
    print("[INFO] Saved → models/best_model.pkl")

    # ── Step 6: Generate plots ──────────────────────────────────────
    print("\n[STEP 6] Generating result plots...")
    plot_model_comparison(results_df)
    plot_roc_curves(models_trained, X_test, y_test)
    plot_confusion_matrix(best_model, best_name, X_test, y_test)

    # ── Step 7: SHAP feature importance ────────────────────────────
    print("\n[STEP 7] Generating SHAP explanation...")
    # Import here so the rest of the script works even if shap isn't installed
    try:
        from src.explain import plot_shap_importance
        plot_shap_importance(best_model, X_test, feature_names, best_name)
    except Exception as e:
        print(f"[WARN] SHAP plot skipped: {e}")

    print("\n" + "=" * 60)
    print("  Pipeline complete! Check results/ for all plots.")
    print("  Next: run  →  streamlit run app/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
