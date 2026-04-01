import streamlit as st

st.title("AI-Based Student Performance and CGPA Prediction System")

study_hours = st.number_input("Study Hours")
attendance = st.number_input("Attendance (%)")
assignment = st.number_input("Assignment Marks (%)")

sem1 = st.number_input("Sem1 GPA")
sem2 = st.number_input("Sem2 GPA")
sem3 = st.number_input("Sem3 GPA")
sem4 = st.number_input("Sem4 GPA")
sem5 = st.number_input("Sem5 GPA")
sem6 = st.number_input("Sem6 GPA")
sem7 = st.number_input("Sem7 GPA")
sem8 = st.number_input("Sem8 GPA")

if st.button("Predict Result"):

    if sem8 >= 2.0:
        st.success("Predicted Result: PASS")
    else:
        st.error("Predicted Result: FAIL")