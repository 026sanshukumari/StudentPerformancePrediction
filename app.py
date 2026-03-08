import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("student_performance_model.pkl")

st.title("🎓 Student Performance Prediction")

studytime = st.slider("Study Time (hours)", 1, 10, 5)
failures = st.slider("Number of past failures", 0, 3, 0)
absences = st.slider("Number of absences", 0, 30, 5)

if st.button("Predict Performance"):
    input_data = pd.DataFrame([[studytime, failures, absences]],
                              columns=["studytime", "failures", "absences"])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Student is likely to PASS")
    else:
        st.error("❌ Student is at RISK of FAILING")
