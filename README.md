# ❤️ Heart Disease Prediction

An end-to-end machine learning project that predicts the likelihood of heart
disease from a patient's clinical measurements, built for the AI/ML
internship final project (Intern ID: ZYNVEX-CERT-1344).

## Problem Statement

Cardiovascular disease is one of the leading causes of death worldwide, and
early risk detection can help doctors intervene sooner. This project builds
a classification model that predicts whether a patient is likely to have
heart disease, based on 13 clinical features (age, cholesterol, blood
pressure, ECG results, etc.), and wraps it in an interactive web app.

## Dataset

- **Source:** UCI Machine Learning Repository — Cleveland Heart Disease dataset
- **Size:** 303 patient records, 13 features + 1 binary target
- **Target:** `0` = no heart disease, `1` = heart disease present

## Project Structure

```
heart-disease-prediction/
├── data/
│   └── heart.csv                # dataset
├── app/
│   └── app.py                   # Streamlit web app
├── model/
│   ├── heart_disease_model.pkl  # trained Random Forest model
│   ├── scaler.pkl               # feature scaler
│   └── feature_names.pkl        # ordered feature list
├── notebooks/
│   └── heart_disease_analysis.ipynb  # full EDA + training notebook (Colab-ready)
├── report/
│   ├── Heart_Disease_Prediction_Report.pdf
│   └── assets/                  # charts & screenshots used in the report
├── train_model.py               # training script (script version of the notebook)
├── requirements.txt
└── README.md
```

## Approach

1. **EDA** — distribution of target classes, correlation heatmap, age vs.
   disease status.
2. **Preprocessing** — train/test split (80/20, stratified), feature scaling
   with `StandardScaler`.
3. **Model comparison** — Logistic Regression, KNN, Decision Tree, Random
   Forest, and SVM were trained and compared using accuracy, precision,
   recall, F1, and ROC-AUC with 5-fold cross-validation.
4. **Hyperparameter tuning** — `GridSearchCV` on the Random Forest (best
   performer) over `n_estimators`, `max_depth`, `min_samples_split`.
5. **Deployment** — the tuned model is served through a Streamlit app where
   a user enters patient details and gets an instant prediction + probability.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Random Forest (tuned) | ~0.82 | ~0.76 | ~0.97 | ~0.85 | ~0.91 |

See `report/assets/model_comparison.csv` and the PDF report for the full
comparison table and charts.

## How to Run Locally

```bash
git clone <your-repo-url>
cd heart-disease-prediction
pip install -r requirements.txt

# Retrain the model (optional — a trained model is already included)
python train_model.py

# Launch the app
streamlit run app/app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## How to Run on Google Colab

Open `notebooks/heart_disease_analysis.ipynb` in Google Colab, upload
`data/heart.csv` when prompted (or mount Google Drive), and run all cells.

## Future Work

- Collect a larger, more diverse patient dataset to improve generalization.
- Add explainability (SHAP values) so predictions can be interpreted per patient.
- Try gradient boosting models (XGBoost/LightGBM) for potentially higher accuracy.
- Add user authentication and store predictions for longitudinal tracking.
- Package the model behind a REST API (FastAPI) for integration with other systems.

## Disclaimer

This project is for educational purposes only and must not be used as a
substitute for professional medical diagnosis or advice.
