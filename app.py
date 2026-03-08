import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load Trained Model
# ---------------------------
model = joblib.load("student_performance_model.pkl")

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="🎓 Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------
# Title & Description
# ---------------------------
st.title("🎓 Student Performance Prediction System")
st.markdown(
    "Predict the probability of failing and assess the student's risk level based on grades, study habits, and attendance."
)

# ---------------------------
# Input Section (3 Columns Layout)
# ---------------------------
st.subheader("Enter Student Details")
col1, col2, col3 = st.columns(3)

with col1:
    G1 = st.slider("G1 Marks (First Exam)", 0, 20, 10)
    studytime = st.selectbox(
        "Weekly Study Time",
        [1,2,3,4],
        help="1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs"
    )

with col2:
    G2 = st.slider("G2 Marks (Second Exam)", 0, 20, 10)
    failures = st.number_input("Past Class Failures", 0, 5, 0)

with col3:
    absences = st.slider("Number of Absences", 0, 50, 5)
    famrel = st.selectbox("Family Relationship Quality", [1,2,3,4,5])
    goout = st.selectbox("Going Out With Friends", [1,2,3,4,5])
    health = st.selectbox("Health Status", [1,2,3,4,5])

# ---------------------------
# Prepare Input
# ---------------------------
student_data = pd.DataFrame([{
    'G1': G1, 'G2': G2, 'studytime': studytime,
    'failures': failures, 'absences': absences,
    'famrel': famrel, 'goout': goout, 'health': health
}])

# ---------------------------
# Prediction & Display
# ---------------------------
def risk_level(prob):
    if prob < 0.3:
        return "LOW RISK ✅"
    elif prob < 0.6:
        return "MEDIUM RISK ⚠️"
    else:
        return "HIGH RISK ❌"

def intervention_action(prob):
    if prob > 0.6:
        return "Immediate counseling + extra classes"
    elif prob > 0.3:
        return "Monitor attendance and provide support"
    else:
        return "No intervention needed"

if st.button("Predict"):
    fail_prob = model.predict_proba(student_data)[0][0]
    
    # Display with metrics and progress bar
    st.subheader(f"📉 Fail Probability: {round(fail_prob, 2)}")
    st.progress(fail_prob)  # progress bar for visual effect
    st.subheader(f"⚠️ Risk Level: {risk_level(fail_prob)}")
    st.subheader(f"💡 Suggested Action: {intervention_action(fail_prob)}")

# ---------------------------
# Optional: Example Student
# ---------------------------
st.markdown("---")
if st.button("Apply Example Student Data"):
    st.info("Example student values applied!")
    example_data = {
        'G1': 12, 'G2': 11, 'studytime': 2,
        'failures': 0, 'absences': 6,
        'famrel': 4, 'goout': 2, 'health': 3
    }
    st.write(example_data)
