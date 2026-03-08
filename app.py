# (Yaha sirf Streamlit ka code hota hai)

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("student_performance_model.pkl")

st.title("🎓 Student Performance Prediction")

# UI inputs
G1 = st.slider("G1 Marks", 0, 20, 10)
G2 = st.slider("G2 Marks", 0, 20, 10)
studytime = st.selectbox("Study Time", [1,2,3,4])
failures = st.selectbox("Past Failures", [0,1,2,3])
absences = st.slider("Absences", 0, 50, 5)
famrel = st.selectbox("Family Relationship", [1,2,3,4,5])
goout = st.selectbox("Going Out", [1,2,3,4,5])
health = st.selectbox("Health", [1,2,3,4,5])

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

if st.button("Predict"):
    prob = model.predict_proba(input_data)[0][0]
    st.write("Fail Probability:", round(prob, 2))
