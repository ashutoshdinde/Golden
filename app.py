import streamlit as st
import pickle
import numpy as np

with open('heart_disease_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.title("Heart Disease Prediction App")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0)
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "True" if x == 1 else "False")
restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, value=0)
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=6.0, value=1.0)
slope = st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3, value=1)

user_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

if st.button("Predict"):
    if user_data.shape[1] != model.n_features_in_:
        st.error("Input data shape does not match model's expected input shape.")
    else:
        prediction = model.predict(user_data)[0]
        if prediction == 1:
            st.error("Warning: You may be at risk for heart disease!")
        else:
            st.success("You are not at risk for heart disease.")


st.write("This app is a basic prediction tool for heart disease based on input metrics.")
