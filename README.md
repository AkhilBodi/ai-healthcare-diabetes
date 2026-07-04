# AI-Powered Diabetes Prediction System
### Exploring the Impact of Artificial Intelligence on Healthcare

**Author:** Sai Saratchandra Phaneendra Akhil  
**University ID:** O24MSD110034  
**Programme:** M.Sc. (Hons.) in Data Science — Chandigarh University  
**Academic Year:** 2025–2026  

---

## Project Overview

This project implements an **AI-powered clinical decision support system** for diabetes risk prediction using the Pima Indians Diabetes Dataset. It demonstrates the practical application of machine learning in healthcare — directly complementing the research study *"Exploring the Impact of Artificial Intelligence on Healthcare"*.

The system:
- Trains and compares **four machine learning models**
- Generates **explainable AI (SHAP) feature importance plots**
- Deploys a **Streamlit web application** for interactive clinical prediction
- Produces **performance visualisations** (ROC curves, confusion matrices)

---

## Clinical Context

Diabetes affects over **537 million adults globally** (IDF, 2021) and is a leading driver of cardiovascular disease, kidney failure, and blindness. Early AI-assisted prediction enables timely clinical intervention — directly addressing the research objectives explored in the accompanying project report.

---

## Dataset

**Source:** Pima Indians Diabetes Database (originally from the National Institute of Diabetes and Digestive and Kidney Diseases)  
**Access:** Built into `sklearn.datasets` — no manual download required  
**Samples:** 768 female patients (age ≥ 21)  
**Features:** 8 clinical variables  
**Target:** Binary (0 = No Diabetes, 1 = Diabetes)

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (2-hr oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Diabetes pedigree function (genetic risk) |
| Age | Age in years |

---

## Project Structure

```
ai-healthcare-diabetes/
│
├── README.md                  ← You are here
├── requirements.txt           ← All Python dependencies
│
├── data/
│   └── diabetes.csv           ← Dataset (auto-generated on first run)
│
├── src/
│   ├── preprocess.py          ← Data loading, cleaning, and splitting
│   ├── train.py               ← Model training and evaluation pipeline
│   └── explain.py             ← SHAP explainability module
│
├── app/
│   └── app.py                 ← Streamlit web application
│
├── models/
│   └── best_model.pkl         ← Saved best model (auto-generated)
│
└── results/
    ├── roc_curve.png          ← ROC comparison plot
    ├── confusion_matrix.png   ← Confusion matrix heatmap
    ├── feature_importance.png ← SHAP feature importance bar chart
    └── model_comparison.png   ← Model accuracy comparison
```

---

## Models Trained

| Model | Type | Key Strength |
|---|---|---|
| Logistic Regression | Linear | Interpretable, fast baseline |
| Decision Tree | Tree-based | Fully explainable, visual |
| Random Forest | Ensemble | High accuracy, handles non-linearity |
| Gradient Boosting | Ensemble | Best performance, robust to outliers |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-healthcare-diabetes.git
cd ai-healthcare-diabetes
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the models
```bash
python src/train.py
```
This will:
- Load and preprocess the dataset
- Train all four models
- Print performance metrics
- Save plots to `results/`
- Save the best model to `models/best_model.pkl`

### 4. Run the web application
```bash
streamlit run app/app.py
```
Opens at `http://localhost:8501` — enter patient values and get an AI prediction with explanation.

---

## Results

After training, results are saved to the `results/` folder:

- **ROC Curve** — Compares all four models visually
- **Confusion Matrix** — True/False Positive/Negative breakdown
- **SHAP Feature Importance** — Which clinical features most influence prediction
- **Model Comparison** — Side-by-side accuracy, precision, recall, F1

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| scikit-learn | ML models, preprocessing, evaluation |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualisation |
| shap | Explainable AI (XAI) |
| streamlit | Web application |
| joblib | Model serialisation |

---

## Connection to Research Report

This project operationalises the findings from the research report *"Exploring the Impact of Artificial Intelligence on Healthcare"* by:

1. **Demonstrating AI predictive analytics** — Mirrors the literature on EHR-based ML prediction (Badawy et al., 2023; Al-Antari et al., 2024)
2. **Implementing Explainable AI** — SHAP values address the governance challenge identified in the report (Raza et al., 2024; McKee & Wouters, 2023)
3. **Providing a HITL interface** — The Streamlit app positions AI as decision support, not autonomous decision-maker, consistent with the report's recommendations
4. **Demonstrating feasibility** — Runs on standard CPU hardware with no GPU, validating the LMIC deployment pathway discussed in Chapter 7

---

## Academic References

- Al-Antari, M. A., et al. (2024). Unveiling the influence of AI predictive analytics on patient outcomes. *Cureus*. PMC11161909.
- Badawy, M., Ramadan, N., & Hefny, H. A. (2023). Healthcare predictive analytics using ML and DL: A survey. *Journal of Electrical Systems and Information Technology*, 10, 40.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Raza, A., et al. (2024). Ethical and regulatory challenges of AI in healthcare. *Heliyon*, 10(4), e26297.

---

## License

MIT License — free to use for academic purposes.

---

*Project submitted as part of the M.Sc. Data Science programme, Chandigarh University, 2025–2026.*
