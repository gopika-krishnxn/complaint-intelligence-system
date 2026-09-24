import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("Complaint Intelligence Dashboard")

st.write(
    "Analyze consumer complaints and explore machine learning model performance."
)

st.divider()


# Load dataset
df = pd.read_csv("data/complaints-2025-01-03_07_55.csv")
model = joblib.load("complaint_model.pkl")


# Live Complaint Classification
st.subheader("Live Complaint Classification")

st.write(
    "Enter a consumer complaint below and the trained machine learning "
    "model will predict its product category."
)

complaint_text = st.text_area(
    "Enter your complaint:",
    placeholder="Example: I was charged an unexpected fee on my credit card."
)


# Priority function
def get_priority(text):
    text = text.lower()

    high_priority_words = [
        "fraud",
        "fraudulent",
        "unauthorized",
        "identity theft",
        "stolen",
        "scam",
        "hack",
        "hacked"
    ]

    medium_priority_words = [
        "fee",
        "charge",
        "payment",
        "billing",
        "interest",
        "refund"
    ]

    for word in high_priority_words:
        if word in text:
            return "High"

    for word in medium_priority_words:
        if word in text:
            return "Medium"

    return "Low"


# Summary function
def get_summary(text):
    sentences = text.split(".")

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1)
    best_sentence_index = sentence_scores.argmax()
    best_sentence = sentences[best_sentence_index]
    return best_sentence + "."

def get_recommended_action(category, priority):

    if priority == "High":
        return "Escalate the complaint for urgent review."

    if category == "Credit card":
        return "Review the credit card transaction and verify the reported charge or payment issue."

    elif category == "Student loan":
        return "Review the student's loan account and verify the reported payment or repayment issue."

    elif category == "Credit reporting or other personal consumer reports":
        return "Review the consumer's credit report and investigate the reported information."

    else:
        return "Review the complaint and investigate the reported issue."

# Classify complaint
if st.button("Classify Complaint"):
    if complaint_text.strip():

        prediction = model.predict([complaint_text])[0]
        decision_scores = model.decision_function([complaint_text])
        scores = decision_scores[0]

        confidence = np.exp(scores) / np.exp(scores).sum()

        predicted_index = np.argmax(scores)

        predicted_confidence = confidence[predicted_index] * 100
        
        priority = get_priority(complaint_text)
        summary = get_summary(complaint_text)
        recommended_action = get_recommended_action(prediction, priority)

        st.success(f"Predicted Category: {prediction}")
        st.metric(
            "Model Confidence",
             f"{predicted_confidence:.2f}%"
        )

        st.info(f"Complaint Summary: {summary}")
        st.info(f"Recommended Action: {recommended_action}")

        if priority == "High":
            st.error(f"Priority Level: {priority}")
        elif priority == "Medium":
            st.warning(f"Priority Level: {priority}")
        else:
            st.success(f"Priority Level: {priority}")

        st.caption(
            "Category prediction uses TF-IDF + Linear SVM. "
            "Priority is determined using complaint keywords."
        )

    else:
        st.warning("Please enter a complaint before classifying.")


# Dataset statistics
total_complaints = len(df)
total_categories = df["Product"].nunique()
training_samples = int(total_complaints * 0.8)
testing_samples = total_complaints - training_samples

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Complaints", total_complaints)
col2.metric("Product Categories", total_categories)
col3.metric("Training Samples", training_samples)
col4.metric("Testing Samples", testing_samples)


# Complaint distribution
st.subheader("Complaint Distribution by Product")

category_counts = df["Product"].value_counts()

st.bar_chart(category_counts)

st.caption(
    "The chart shows the number of consumer complaints in each product category."
)


# Model performance
st.subheader("Model Performance")

with open("model_accuracy.txt", "r") as file:
    model_accuracy = float(file.read())

st.metric(
    "SVM Accuracy",
    f"{model_accuracy * 100:.2f}%"
)

st.caption(
    "Accuracy represents the percentage of test complaints correctly "
    "classified by the SVM model."
)


# Model information
st.subheader("Model Information")

info1, info2, info3 = st.columns(3)

info1.write("**Algorithm**")
info1.write("Linear Support Vector Machine (SVM)")

info2.write("**Feature Extraction**")
info2.write("TF-IDF")

info3.write("**Class Weighting**")
info3.write("Balanced")


# Confusion matrix
st.subheader("Confusion Matrix")

matrix = np.loadtxt(
    "confusion_matrix.txt",
    dtype=int
)

labels = [
    "Credit Card",
    "Credit Reporting",
    "Student Loan"
]

fig, ax = plt.subplots()

sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels,
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

st.caption(
    "The confusion matrix shows how accurately the model classified "
    "complaints across the three product categories."
)