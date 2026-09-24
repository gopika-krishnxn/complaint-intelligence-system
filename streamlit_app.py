import streamlit as st

st.title("Complaint Intelligence System")

st.write(
    "An AI-powered system that analyzes consumer complaints "
    "using machine learning and natural language processing."
)

st.divider()

st.subheader("Welcome")

st.write(
    "This system helps analyze consumer complaints by predicting "
    "their product category, identifying priority, and generating "
    "a concise complaint summary."
)

st.subheader("How It Works")

st.write(
    "Enter a complaint in the Dashboard to receive an automated "
    "analysis using the trained machine learning model."
)

st.info(
    "Go to the Dashboard to analyze a complaint."
)

if st.button("Go to Dashboard"):
    st.switch_page("pages/1_Dashboard.py")