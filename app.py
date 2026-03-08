import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_performance_model.pkl")

st.title("🎓 Student Performance Prediction")
st.header("Enter student details:")

# Input fields
G1 = st.number_input("G1 (First Period Grade)", 0, 20, 10)
G2 = st.number_input("G2 (Second Period Grade)", 0, 20, 10)
studytime = st.number_input("Weekly Study Time (1-4)", 1, 4, 2)
failures = st.number_input("Past Failures", 0, 10, 0)
absences = st.number_input("Number of Absences", 0, 100, 5)
famrel = st.number_input("Family Relationship (1-5)", 1, 5, 3)
goout = st.number_input("Going Out Frequency (1-5)", 1, 5, 2)
health = st.number_input("Health Status (1-5)", 1, 5, 3)

# Prediction
student_data = pd.DataFrame([{
    'G1': G1, 'G2': G2, 'studytime': studytime,
    'failures': failures, 'absences': absences,
    'famrel': famrel, 'goout': goout, 'health': health
}])

def risk_level(prob):
    if prob < 0.3:
        return "LOW RISK"
    elif prob < 0.6:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"

def intervention_action(prob):
    if prob > 0.6:
        return "Immediate counseling + extra classes"
    elif prob > 0.3:
        return "Monitor attendance and provide support"
    else:
        return "No intervention needed"

if st.button("Predict"):
    fail_prob = model.predict_proba(student_data)[0][0]
    st.subheader(f"📉 Fail Probability: {round(fail_prob,2)}")
    st.subheader(f"⚠️ Risk Level: {risk_level(fail_prob)}")
    st.subheader(f"💡 Suggested College Action: {intervention_action(fail_prob)}")
