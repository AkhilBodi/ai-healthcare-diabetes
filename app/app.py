"""
app.py
------
Streamlit web application for the AI Diabetes Prediction System.

Provides:
  - Interactive patient data input (sliders)
  - Real-time prediction with confidence score
  - SHAP feature importance explanation per patient
  - Summary of all model performance metrics

Run from the project root with:
    streamlit run app/app.py
"""

import os
import sys
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Make sure src/ is importable from the app folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Diabetes Prediction | Healthcare AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS: clean clinical look ──────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0A1628, #0D2137);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: #0EA5E9; margin: 0; font-size: 2rem; }
    .main-header p  { color: #BAE6FD; margin: 0.5rem 0 0 0; font-size: 1rem; }

    .result-positive {
        background: #FEF2F2; border: 2px solid #DC2626;
        border-radius: 10px; padding: 1.2rem; margin: 1rem 0;
    }
    .result-negative {
        background: #F0FDF4; border: 2px solid #059669;
        border-radius: 10px; padding: 1.2rem; margin: 1rem 0;
    }
    .metric-card {
        background: #EFF6FF; border: 1px solid #BFDBFE;
        border-radius: 8px; padding: 1rem; text-align: center;
    }
    .shap-positive { color: #DC2626; font-weight: bold; }
    .shap-negative { color: #059669; font-weight: bold; }
    .footer { color: #64748B; font-size: 0.85rem; text-align: center; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the saved best model. Run train.py first if this fails."""
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        st.error(
            "❌ Model file not found. Please run `python src/train.py` first to train and save the model."
        )
        st.stop()
    return joblib.load(model_path)


bundle       = load_model()
model        = bundle["model"]
scaler       = bundle["scaler"]
feature_names = bundle["features"]
model_name   = bundle["name"]


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 AI Diabetes Risk Prediction System</h1>
    <p>Exploring the Impact of Artificial Intelligence on Healthcare &nbsp;|&nbsp;
       M.Sc. Data Science, Chandigarh University &nbsp;|&nbsp;
       Sai Saratchandra Phaneendra Akhil (O24MSD110034)</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
> **Active Model:** `{model_name}` &nbsp;|&nbsp;
  **Dataset:** Pima Indians Diabetes (768 patients, 8 clinical features) &nbsp;|&nbsp;
  **Purpose:** Clinical decision support — AI assists, clinician decides.
""")


# ── Sidebar: Patient Input ────────────────────────────────────────────────────
st.sidebar.header("👤 Patient Clinical Data")
st.sidebar.markdown("Enter the patient's clinical measurements below:")

# Each slider has: (label, min, max, default, step, help text)
inputs = {
    "Pregnancies":             st.sidebar.slider("Pregnancies",                 0,   17,   3,    1,    "Number of times pregnant"),
    "Glucose":                 st.sidebar.slider("Glucose (mg/dL)",            50,  200,  120,   1,    "Plasma glucose — 2-hr oral glucose test"),
    "BloodPressure":           st.sidebar.slider("Blood Pressure (mm Hg)",     30,  122,  70,    1,    "Diastolic blood pressure"),
    "SkinThickness":           st.sidebar.slider("Skin Thickness (mm)",         0,   99,  23,    1,    "Triceps skin fold thickness"),
    "Insulin":                 st.sidebar.slider("Insulin (mu U/ml)",           0,  846,  79,    1,    "2-hour serum insulin"),
    "BMI":                     st.sidebar.slider("BMI (kg/m²)",               10.0, 67.1, 32.0, 0.1,  "Body mass index"),
    "DiabetesPedigreeFunction":st.sidebar.slider("Diabetes Pedigree Function",0.05, 2.42, 0.47, 0.01, "Genetic diabetes risk score"),
    "Age":                     st.sidebar.slider("Age (years)",                21,   81,  33,    1,    "Patient age"),
}

predict_button = st.sidebar.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**📌 Clinical Note:**
This tool is a *decision support aid* only.
All predictions must be reviewed and confirmed
by a qualified clinician. AI does not replace
clinical judgement.
""")


# ── Main content: 3 columns ───────────────────────────────────────────────────
col1, col2, col3 = st.columns([1.1, 1.2, 1.1])

with col1:
    st.subheader("📋 Patient Summary")
    # Display entered values as a neat table
    df_display = {"Feature": list(inputs.keys()), "Value": list(inputs.values())}
    st.dataframe(df_display, use_container_width=True, hide_index=True)

with col2:
    st.subheader("🤖 AI Prediction")

    if predict_button:
        # ── Build input vector ──────────────────────────────────────
        input_array  = np.array([[inputs[f] for f in feature_names]])
        input_scaled = scaler.transform(input_array)

        # ── Get prediction and probability ──────────────────────────
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0]
        risk_pct     = probability[1] * 100

        # ── Display result ──────────────────────────────────────────
        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                <h3 style="color:#DC2626; margin:0;">⚠️ High Diabetes Risk Detected</h3>
                <p style="margin:0.5rem 0 0 0; font-size:1.1rem;">
                    Predicted Risk Probability: <strong>{risk_pct:.1f}%</strong>
                </p>
                <p style="color:#7F1D1D; margin:0.4rem 0 0 0; font-size:0.9rem;">
                    Recommendation: Refer for confirmatory HbA1c and fasting glucose tests.
                    Lifestyle intervention counselling advised.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <h3 style="color:#059669; margin:0;">✅ Low Diabetes Risk</h3>
                <p style="margin:0.5rem 0 0 0; font-size:1.1rem;">
                    Predicted Risk Probability: <strong>{risk_pct:.1f}%</strong>
                </p>
                <p style="color:#064E3B; margin:0.4rem 0 0 0; font-size:0.9rem;">
                    Recommendation: Routine follow-up. Monitor BMI, glucose, and blood pressure annually.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Confidence gauge ────────────────────────────────────────
        st.markdown("**Prediction Confidence:**")
        col_a, col_b = st.columns(2)
        col_a.metric("No Diabetes",  f"{probability[0]*100:.1f}%")
        col_b.metric("Diabetes",     f"{probability[1]*100:.1f}%",
                     delta=f"{'⬆ Risk' if prediction==1 else '⬇ Low risk'}")

        # ── SHAP Explanation ────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Why this prediction? (SHAP Explanation)")
        st.markdown("""
        SHAP values show which features **increased** (🔴) or **decreased** (🟢)
        the predicted diabetes risk for this specific patient.
        """)

        try:
            from src.explain import explain_single_patient
            shap_dict = explain_single_patient(model, scaler, feature_names, inputs)

            # Plot horizontal bar chart for SHAP values
            fig, ax = plt.subplots(figsize=(6, 4))
            features_sorted = list(shap_dict.keys())
            values_sorted   = list(shap_dict.values())
            bar_colors = ["#DC2626" if v > 0 else "#059669" for v in values_sorted]
            ax.barh(features_sorted, values_sorted, color=bar_colors, alpha=0.85)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP Value (positive = increases diabetes risk)")
            ax.set_title("Feature Contributions for This Patient", fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")

    else:
        st.info("👈 Adjust patient values in the sidebar and click **Predict Diabetes Risk**.")

with col3:
    st.subheader("📊 Model Performance")

    # Show stored results plots if they exist
    if os.path.exists("results/model_comparison.png"):
        st.image("results/model_comparison.png", caption="All Models — Metric Comparison")
    if os.path.exists("results/roc_curve.png"):
        st.image("results/roc_curve.png", caption="ROC Curves — All Models")
    if os.path.exists("results/feature_importance.png"):
        st.image("results/feature_importance.png", caption="SHAP Feature Importance (Test Set)")
    if not os.path.exists("results/model_comparison.png"):
        st.info("Run `python src/train.py` to generate performance charts.")


# ── Bottom section: About ─────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ About this Application"):
    st.markdown(f"""
    **AI Diabetes Prediction System** — Project component for the MSc Data Science research study:
    *"Exploring the Impact of Artificial Intelligence on Healthcare"*

    | | |
    |---|---|
    | **Active Model** | {model_name} |
    | **Dataset** | Pima Indians Diabetes Database (UCI / OpenML) |
    | **Training / Test Split** | 80% / 20% (stratified) |
    | **Preprocessing** | StandardScaler normalisation; zero-value imputation with median |
    | **Explainability** | SHAP (SHapley Additive exPlanations) — Lundberg & Lee, 2017 |
    | **AI Positioning** | Decision Support (Human-in-the-Loop) — clinician retains final authority |

    **Clinical Disclaimer:** This application is a research demonstration built for academic purposes.
    It is NOT a certified medical device and must NOT be used for actual clinical diagnosis.
    """)

st.markdown("""
<div class="footer">
    Sai Saratchandra Phaneendra Akhil &nbsp;|&nbsp; O24MSD110034 &nbsp;|&nbsp;
    M.Sc. Data Science (Hons.) &nbsp;|&nbsp; Chandigarh University &nbsp;|&nbsp; 2025–2026
</div>
""", unsafe_allow_html=True)
