# app_full.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI CGPA Prediction", layout="wide")
st.title("AI-Based Student CGPA Prediction System")

# -----------------------------
# Step 1: Dataset
# -----------------------------
data = {
    'Study Hours':[2,3,4,5,6],
    'Attendance':[60,70,80,90,85],
    'Assignment':[50,60,70,80,90],
    'Sem1':[2.1,2.4,2.8,3.0,3.2],
    'Sem2':[2.2,2.5,2.9,3.1,3.3],
    'Sem3':[2.3,2.6,3.0,3.2,3.4],
    'Sem4':[2.4,2.7,3.1,3.3,3.5],
    'Sem5':[2.5,2.8,3.2,3.4,3.6],
    'Sem6':[2.6,2.9,3.3,3.5,3.7],
    'Sem7':[2.7,3.0,3.4,3.6,3.8],
    'Sem8':[2.8,3.1,3.5,3.7,3.9],
    'Final CGPA':[2.5,2.8,3.2,3.4,3.6]
}

df = pd.DataFrame(data)

# -----------------------------
# Step 2: ML Model Training
# -----------------------------
X = df[['Study Hours','Attendance','Assignment','Sem1','Sem2','Sem3','Sem4','Sem5','Sem6','Sem7','Sem8']]
y = df['Final CGPA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Step 3: User Input
# -----------------------------
st.sidebar.header("Enter Student Details")

study_hours = st.sidebar.number_input("Study Hours", 0, 24, 4)
attendance = st.sidebar.number_input("Attendance (%)", 0, 100, 85)
assignment = st.sidebar.number_input("Assignment (%)", 0, 100, 70)
sem1 = st.sidebar.number_input("Semester 1 CGPA", 0.0, 4.0, 2.6)
sem2 = st.sidebar.number_input("Semester 2 CGPA", 0.0, 4.0, 2.8)
sem3 = st.sidebar.number_input("Semester 3 CGPA", 0.0, 4.0, 3.0)
sem4 = st.sidebar.number_input("Semester 4 CGPA", 0.0, 4.0, 3.1)
sem5 = st.sidebar.number_input("Semester 5 CGPA", 0.0, 4.0, 3.2)
sem6 = st.sidebar.number_input("Semester 6 CGPA", 0.0, 4.0, 3.3)
sem7 = st.sidebar.number_input("Semester 7 CGPA", 0.0, 4.0, 3.4)
sem8 = st.sidebar.number_input("Semester 8 CGPA", 0.0, 4.0, 3.5)

student_input = [[study_hours, attendance, assignment, sem1, sem2, sem3, sem4, sem5, sem6, sem7, sem8]]

# -----------------------------
# Step 4: Prediction
# -----------------------------
predicted_cgpa = model.predict(student_input)[0]
pass_fail = "PASS" if predicted_cgpa >= 2.0 else "FAIL"

st.subheader("Predicted Results")
st.write(f"**Predicted Final CGPA:** {predicted_cgpa:.2f}")
st.write(f"**Result:** {pass_fail}")

# -----------------------------
# Step 5: Graphs
# -----------------------------
st.subheader("Graphs")

# 1️⃣ Actual vs Predicted CGPA (test set)
y_pred_test = model.predict(X_test)
fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred_test, color='blue')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax1.set_xlabel("Actual CGPA")
ax1.set_ylabel("Predicted CGPA")
ax1.set_title("Actual vs Predicted CGPA (Test Set)")
st.pyplot(fig1)

# 2️⃣ Feature Importance
importance = model.coef_
features = X.columns
fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.bar(features, importance, color='skyblue')
ax2.set_xticklabels(features, rotation=45, ha='right')
ax2.set_ylabel("Coefficient Value")
ax2.set_title("Feature Importance for CGPA Prediction")
st.pyplot(fig2)