import streamlit as st

st.title("About the Project")

st.write(
    "Complaint Intelligence System is a machine learning-based application "
    "that classifies consumer complaints into Credit Card, Credit Reporting, "
    "and Student Loan categories."
)
st.header("Project Overview")

st.write(
    "The system uses Natural Language Processing (NLP) and Machine Learning "
    "to analyze consumer complaint text and predict the complaint category. "
    "The current system classifies complaints into Credit Card, "
    "Credit Reporting, and Student Loan categories."
)
st.header("How It Works")

st.write(
    "The system follows a simple machine learning workflow. "
    "Consumer complaint text is first preprocessed and converted into "
    "numerical features using TF-IDF. The trained machine learning model "
    "then analyzes these features and predicts the complaint category."
)
st.header("Key Features")

st.write(
    """
    • Consumer complaint text analysis  
    • Text preprocessing using NLP techniques  
    • TF-IDF based feature extraction  
    • Machine learning-based complaint classification  
    • Classification into Credit Card, Credit Reporting, and Student Loan categories  
    • Interactive Streamlit interface
    """
)
st.header("Technologies Used")

st.write(
    """
    • Python  
    • Pandas  
    • Scikit-learn  
    • Natural Language Processing (NLP)  
    • TF-IDF  
    • Streamlit  
    • Matplotlib  
    • Seaborn
    """
)
st.header("Project Purpose")

st.write(
    "The purpose of this project is to demonstrate how Natural Language "
    "Processing and Machine Learning can be applied to real-world consumer "
    "complaint data. The system provides a simple way to analyze complaint "
    "text and automatically identify its category."
)