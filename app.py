# app.py - Professional Student Performance Prediction
import streamlit as st
import pandas as pd
import joblib

# Load Trained Model

model = joblib.load("student_performance_model.pkl")

# Page Title & Description

st.set_page_config(
    page_title="🎓 Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction System")
st.markdown(
    "Enter student details below to predict the probability of failing the course and risk level."
)

# Input Section

st.subheader("Student Details")
col1, col2, col3 = st.columns(3)

with col1:
    G1 = st.slider("G1 Marks (First Exam)", 0, 20, 10)
    studytime = st.selectbox(
        "Study Time (Weekly)",
        [1,2,3,4],
        help="1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs"
    )

with col2:
    G2 = st.slider("G2 Marks (Second Exam)", 0, 20, 10)
    failures = st.selectbox(
        "Past Class Failures",
        [0,1,2,3],
        help="Number of times the student failed previous classes"
    )

with col3:
    absences = st.slider("Number of Absences", 0, 50, 5)
    famrel = st.selectbox(
        "Family Relationship Quality",
        [1,2,3,4,5],
        help="1 = Very Bad, 5 = Excellent"
    )
    goout = st.selectbox(
        "Going Out With Friends",
        [1,2,3,4,5],
        help="1 = Very Low, 5 = Very High"
    )
    health = st.selectbox(
        "Health Status",
        [1,2,3,4,5],
        help="1 = Very Bad, 5 = Very Good"
    )

# Prepare Input Data

input_data = pd.DataFrame([{
    'G1': G1,
    'G2': G2,
    'studytime': studytime,
    'failures': failures,
    'absences': absences,
    'famrel': famrel,
    'goout': goout,
    'health': health
}])

# Prediction & Risk Evaluation

if st.button("Predict Performance"):
    
    # Predict fail probability
    fail_prob = model.predict_proba(input_data)[0][0]
    st.write("### Fail Probability:", round(fail_prob, 2))
    
    # Progress bar
    st.progress(fail_prob)
    
    # Risk Level
    if fail_prob < 0.3:
        st.success("LOW RISK ✅")
        st.write("No intervention needed.")
    elif fail_prob < 0.6:
        st.warning("MEDIUM RISK ⚠️")
        st.write("Monitor attendance and provide academic support.")
    else:
        st.error("HIGH RISK ❌")
        st.write("Immediate counseling and extra classes recommended.")

# Sample Student Demo

st.markdown("---")
if st.button("Use Example Student"):
    st.info("Example student data applied!")
    st.session_state.G1 = 12
    st.session_state.G2 = 11
    st.session_state.studytime = 2
    st.session_state.failures = 0
    st.session_state.absences = 6
    st.session_state.famrel = 4
    st.session_state.goout = 2
    st.session_state.health = 3
