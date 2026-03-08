import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("student_performance_model.pkl")

st.set_page_config(page_title="Student Performance Prediction", layout="centered")

st.title("🎓 Student Performance Prediction System")
st.write("Predict whether a student is at risk of failing **before final exams**.")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Student Details")

G1 = st.number_input("Mid-Sem Exam Score (G1)", min_value=0, max_value=20, value=10)
G2 = st.number_input("Pre-Final Exam Score (G2)", min_value=0, max_value=20, value=10)
studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4])
absences = st.number_input("Number of Absences", min_value=0, max_value=100, value=5)
failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Performance"):
    input_data = pd.DataFrame([{
        "G1": G1,
        "G2": G2,
        "studytime": studytime,
        "absences": absences,
        "failures": failures
    }])

    prediction = model.predict(input_data)[0]

    if prediction >= 10:
        st.success("✅ Prediction: PASS")
        st.info("Risk Level: LOW RISK")
    else:
        st.error("❌ Prediction: FAIL")
        st.warning("Risk Level: HIGH RISK")

    st.caption("This prediction helps teachers take early action.")
