import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# -----------------------------
# Load trained model (Cloud-safe path)
# -----------------------------
model_path = os.path.join(os.path.dirname(__file__), "student_performance_model.pkl")
model = joblib.load(model_path)

# -----------------------------
# Sidebar info
# -----------------------------
st.sidebar.title("About the App")
st.sidebar.info("""
**Student Performance Prediction**

- Predicts risk of failing using academic grades, study habits, attendance, and health.
- Model: RandomForestClassifier (robust, avoids overfitting)
- Output: Risk Level + Suggested College Action
""")

# -----------------------------
# Page title
# -----------------------------
st.title("🎓 Student Performance Prediction")
st.subheader("Predict risk and suggest interventions for students")

# -----------------------------
# Input fields in columns
# -----------------------------
col1, col2, col3 = st.columns(3)
G1 = col1.number_input("G1 (First Period Grade)", 0, 20, 10)
G2 = col2.number_input("G2 (Second Period Grade)", 0, 20, 10)
studytime = col3.number_input("Weekly Study Time (1-4)", 1, 4, 2)

col4, col5, col6 = st.columns(3)
failures = col4.number_input("Number of Past Failures", 0, 10, 0)
absences = col5.number_input("Number of Absences", 0, 100, 5)
famrel = col6.number_input("Family Relationship (1-5)", 1, 5, 3)

col7, col8 = st.columns(2)
goout = col7.number_input("Going Out Frequency (1-5)", 1, 5, 2)
health = col8.number_input("Health Status (1-5)", 1, 5, 3)

# -----------------------------
# Risk & Action functions
# -----------------------------
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

# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict"):
    student_data = pd.DataFrame([{
        'G1': G1,
        'G2': G2,
        'studytime': studytime,
        'failures': failures,
        'absences': absences,
        'famrel': famrel,
        'goout': goout,
        'health': health
    }])

    fail_prob = model.predict_proba(student_data)[0][0]
    st.subheader(f"📉 Fail Probability: {round(fail_prob*100,1)}%")

    # -----------------------------
    # Conditional Risk Display
    # -----------------------------
    if fail_prob >= 0.3:  # Only show risk & intervention if moderate/high
        st.subheader(f"⚠️ Risk Level: {risk_level(fail_prob)}")
        st.subheader(f"💡 Suggested College Action: {intervention_action(fail_prob)}")

        # -----------------------------
        # Feature Importance Chart
        # -----------------------------
        feature_names = ['G1','G2','studytime','failures','absences','famrel','goout','health']
        importance = pd.Series(model.feature_importances_, index=feature_names).sort_values()
        st.subheader("📊 Top Factors Affecting Student Performance")
        plt.figure(figsize=(6,4))
        importance.plot(kind='barh', color='skyblue')
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        st.pyplot(plt)
    else:
        st.success("✅ Low Risk — No intervention needed")
