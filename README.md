# AI Complaint Intelligence & Resolution Assistant

An AI/ML-powered application that analyzes consumer complaints, predicts their product category, summarizes the complaint, determines its priority level, and provides a recommended action.

## Overview

The AI Complaint Intelligence & Resolution Assistant is a machine learning-based application designed to analyze consumer complaints and automatically identify the relevant financial product category.

The system uses Natural Language Processing (NLP) techniques to process complaint text and a Linear Support Vector Machine (LinearSVC) model to classify complaints into three categories: Credit Card, Credit Reporting or Other Personal Consumer Reports, and Student Loan.

In addition to classification, the application provides a complaint summary, priority level, and recommended action through an interactive Streamlit dashboard.

## Problem Statement

Consumer complaint datasets contain large amounts of unstructured text, making it difficult to manually identify the type of financial product involved in each complaint.

This project aims to automate the initial analysis of consumer complaints by using machine learning and Natural Language Processing (NLP) to classify complaint text into relevant product categories and provide additional insights such as priority and recommended action.

## Objectives

* Automatically classify consumer complaints into relevant financial product categories.
* Apply Natural Language Processing (NLP) techniques to extract useful information from complaint text.
* Use TF-IDF to convert text data into numerical features for machine learning.
* Build a Linear Support Vector Machine (LinearSVC) classification model.
* Provide a confidence-style indication for the predicted category.
* Generate a concise summary of the complaint.
* Identify complaint priority using rule-based keyword detection.
* Provide a recommended action based on the predicted category and priority.
* Present the results through an interactive Streamlit dashboard.

## Key Features

* **Complaint Classification** — Predicts the financial product category from complaint text.
* **Confidence-Style Score** — Displays an indication of how strongly the model favors the predicted category based on LinearSVC decision scores.
* **Complaint Summarization** — Extracts the most relevant sentence from the complaint.
* **Priority Detection** — Assigns High, Medium, or Low priority using predefined complaint keywords.
* **Recommended Action** — Suggests an appropriate action based on the predicted category and priority.
* **Interactive Dashboard** — Provides live complaint analysis through a Streamlit interface.
* **Dataset Statistics** — Displays complaint counts, product categories, and training/testing sample information.
* **Model Evaluation** — Includes accuracy and a confusion matrix to evaluate classification performance.

## Machine Learning Approach

The project follows a text classification pipeline:

1. **Data Loading** — Consumer complaint data is loaded from the CFPB dataset.
2. **Text Preparation** — The consumer complaint narrative is used as the input text.
3. **Train-Test Split** — The dataset is divided into training and testing sets using an 80/20 split with stratification.
4. **TF-IDF Feature Extraction** — Text is converted into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).
5. **Classification** — A Linear Support Vector Machine (LinearSVC) is trained to classify complaints into three product categories.
6. **Class Balancing** — Balanced class weights are used to help handle the imbalance between product categories.
7. **Evaluation** — The model is evaluated using test-set accuracy and a confusion matrix.
8. **Prediction** — The trained pipeline is used to classify new complaint text entered through the Streamlit dashboard.

## Dataset

The project uses consumer complaint data published by the **Consumer Financial Protection Bureau (CFPB)**.

The dataset contains consumer complaint narratives along with information about the financial products associated with those complaints.

For this project, the following three product categories are used:

* Credit Card
* Credit Reporting or Other Personal Consumer Reports
* Student Loan

The model uses the **Consumer complaint narrative** column as the input text and the **Product** column as the target variable.

The dataset contains **16,428 complaints** used for this project.

## Technologies Used

### Programming Language

* Python

### Machine Learning & NLP

* Scikit-learn
* TF-IDF
* Linear Support Vector Machine (LinearSVC)

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Application & Model Management

* Streamlit
* Joblib

### Development Tools

* Cursor
* Git
* GitHub

## Model Performance

The LinearSVC model was evaluated on a held-out test set containing 20% of the dataset.

* **Test Set Accuracy:** 98.05%
* **Feature Extraction:** TF-IDF
* **Classification Algorithm:** Linear Support Vector Machine (LinearSVC)
* **Class Weighting:** Balanced

The project also includes a confusion matrix to visualize the model's classification performance across the three product categories.

> **Note:** The confidence displayed in the application is a confidence-style score derived from LinearSVC decision scores. It is not a true probability.

## Project Structure

```text
complaint-intelligence-system/
│
├── data/
│   └── complaints-2025-01-03_07_55.csv   # Local dataset, not uploaded to GitHub
│
├── pages/
│   ├── 1_Dashboard.py
│   └── 2_About.py
│
├── app.py
├── streamlit_app.py
├── complaint_model.pkl
├── confusion_matrix.txt
├── model_accuracy.txt
├── .gitignore
└── README.md
```

### Main Files

* `app.py` — Trains the machine learning model and evaluates its performance.
* `streamlit_app.py` — Main Streamlit application page.
* `pages/1_Dashboard.py` — Interactive complaint analysis and model performance dashboard.
* `pages/2_About.py` — Project information page.
* `complaint_model.pkl` — Saved trained machine learning pipeline.
* `model_accuracy.txt` — Stores the model's test accuracy.
* `confusion_matrix.txt` — Stores the confusion matrix values.
* `.gitignore` — Specifies files and folders that should not be uploaded to GitHub.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd complaint-intelligence-system
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn streamlit
```

### 3. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

### 4. Use the Dashboard

1. Open the **Complaint Intelligence System** home page.
2. Navigate to the **Dashboard**.
3. Enter a consumer complaint.
4. Click **Classify Complaint**.
5. View the predicted category, confidence-style score, complaint summary, priority level, and recommended action.

## Example

### Input

> I was charged a late payment fee on my credit card even though I made the payment before the due date.

### Output

* **Predicted Category:** Credit Card
* **Confidence-Style Score:** 94.88%
* **Complaint Summary:** I was charged a late payment fee on my credit card even though I made the payment before the due date.
* **Priority Level:** Medium
* **Recommended Action:** Review the credit card transaction and verify the reported charge or payment issue.

The application displays these results through the Streamlit dashboard.

## Future Improvements

* Improve classification of underrepresented complaint categories by using a larger and more balanced dataset.
* Explore advanced NLP and transformer-based models for improved text classification.
* Improve complaint summarization using dedicated NLP summarization techniques.
* Replace the rule-based priority system with a machine learning-based priority classification model.
* Add more complaint categories as additional data becomes available.
* Deploy the application as a publicly accessible web application.

## Author

**Gopika Krishnan**

B.E. Computer Science and Engineering student with a focus on **AI/ML Engineering**.

Interested in building practical machine learning applications using Python, NLP, and modern AI/ML technologies.
