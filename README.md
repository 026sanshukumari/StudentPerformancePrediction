# 🎓 Student Performance Prediction System

**Description:**  
An interactive web app to predict student exam performance and risk of failure using a trained Random Forest model. The system provides fail probability, risk level, and recommended interventions.

---

## **Features**
- Predict PASS/FAIL based on exam scores, study effort, attendance, and personal factors  
- Displays **Fail Probability** and **Risk Level (LOW/MEDIUM/HIGH)**  
- Suggests appropriate **intervention actions**  
- Optional **visualizations** for data analysis  

---

## **Technologies**
Python | Pandas | NumPy | Scikit-learn | Joblib | Streamlit | Matplotlib | Seaborn  

---

## **Project Files**
- `app.py` – Streamlit web application  
- `requirements.txt` – Python dependencies  
- `student_performance_model.pkl` – Trained ML model  
- `student-mat.csv` – Dataset  
- `README.md` – Project documentation  

---

## **Usage**
```bash
# Clone repository
git clone https://github.com/<your-username>/StudentPerformancePrediction.git
cd StudentPerformancePrediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
