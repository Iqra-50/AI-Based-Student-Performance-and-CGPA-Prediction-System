# app_full.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ----------------------------
# Step 1: Prepare dataset
# ----------------------------
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

# ----------------------------
# Step 2: Train Linear Regression Model
# ----------------------------
X = df[['Study Hours','Attendance','Assignment','Sem1','Sem2','Sem3','Sem4','Sem5','Sem6','Sem7','Sem8']]
y = df['Final CGPA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# Step 3: Streamlit UI
# ----------------------------
st.title("AI-Based Student CGPA Prediction")

# Sidebar: Student Inputs
st.sidebar.header("Enter Student Details")
study_hours = st.sidebar.number_input("Study Hours", 0, 24, 4)
attendance = st.sidebar.number_input("Attendance %", 0, 100, 85)
assignment = st.sidebar.number_input("Assignment Score", 0, 100, 70)
sem1 = st.sidebar.number_input("Sem1 CGPA", 0.0, 4.0, 2.6)
sem2 = st.sidebar.number_input("Sem2 CGPA", 0.0, 4.0, 2.8)
sem3 = st.sidebar.number_input("Sem3 CGPA", 0.0, 4.0, 3.0)
sem4 = st.sidebar.number_input("Sem4 CGPA", 0.0, 4.0, 3.1)
sem5 = st.sidebar.number_input("Sem5 CGPA", 0.0, 4.0, 3.2)
sem6 = st.sidebar.number_input("Sem6 CGPA", 0.0, 4.0, 3.3)
sem7 = st.sidebar.number_input("Sem7 CGPA", 0.0, 4.0, 3.4)
sem8 = st.sidebar.number_input("Sem8 CGPA", 0.0, 4.0, 3.5)

student = [[study_hours, attendance, assignment, sem1, sem2, sem3, sem4, sem5, sem6, sem7, sem8]]
predicted_cgpa = model.predict(student)[0]

# ----------------------------
# Step 4: Display predicted results
# ----------------------------
st.subheader("Predicted Final CGPA")
st.write(round(predicted_cgpa,2))

st.subheader("Result")
if predicted_cgpa >= 2.0:
    st.success("PASS ✅")
else:
    st.error("FAIL ❌")

# ----------------------------
# Step 5: Graphs
# ----------------------------

# 1. Actual vs Predicted scatter plot (dataset + input student)
pred_all = model.predict(X)
fig1, ax1 = plt.subplots()
ax1.scatter(y, pred_all, color='red', label='Dataset Predictions')  # dataset points
ax1.scatter(predicted_cgpa, predicted_cgpa, color='blue', s=100, label='Your Input')  # single student
ax1.plot([min(y), max(y)], [min(y), max(y)], color='black', linestyle='--', label='Ideal Line')
ax1.set_xlabel("Actual CGPA")
ax1.set_ylabel("Predicted CGPA")
ax1.set_title("Actual vs Predicted CGPA")
ax1.legend()
st.pyplot(fig1)

# 2. Feature Importance
importance = model.coef_
features = X.columns
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.bar(features, importance)
ax2.set_xticks(range(len(features)))
ax2.set_xticklabels(features, rotation=45)
ax2.set_title("Feature Importance for CGPA Prediction")
st.pyplot(fig2)

# 3. Pass/Fail Pie chart
fig3, ax3 = plt.subplots()
results = ['Pass', 'Fail']
counts = [1 if predicted_cgpa >=2 else 0, 1 if predicted_cgpa <2 else 0]
ax3.pie(counts, labels=results, autopct='%1.1f%%', colors=['green','red'])
ax3.set_title("Predicted Result")
st.pyplot(fig3)
