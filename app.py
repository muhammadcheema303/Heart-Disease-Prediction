import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('heart_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to predict the likelihood of heart disease.")

# Input fields for 13 features
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.selectbox("Chest Pain Type (cp: 0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 90, 200, 120)
chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of the peak exercise ST segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-4) colored by fluoroscopy", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2])

if st.button("Predict Heart Disease"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error("⚠️ Warning: The model predicts a **High Risk** of Heart Disease.")
    else:
        st.success("✅ Good News: The model predicts a **Low Risk** of Heart Disease.")
