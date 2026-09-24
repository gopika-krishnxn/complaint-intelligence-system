import pandas as pd
import string
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )),
    ("svm", LinearSVC(class_weight="balanced"))
])


df = pd.read_csv("data/complaints-2025-01-03_07_55.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df["Consumer complaint narrative"].iloc[0])
text = df["Consumer complaint narrative"].iloc[0]
text = text.lower()
text = text.translate(str.maketrans("", "", string.punctuation))
print(text)
tokens = text.split()
print(tokens)


X = df["Consumer complaint narrative"].fillna("")

y = df["Product"]

print("Target column:")
print(y.head())

print("\nNumber of unique products:", y.nunique())
print("\nProduct categories:")
print(y.unique())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

pipeline.fit(X_train, y_train)
print("\nSVM model training completed!")

joblib.dump(pipeline, "complaint_model.pkl")
print("\nModel saved successfully!")

y_pred = pipeline.predict(X_test)
print("\nFirst 10 predictions:")
print(y_pred[:10])
print("\nActual first 10:")
print(y_test.iloc[:10].values)

print("\nPredicted first 10:")
print(y_pred[:10])
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

with open("model_accuracy.txt", "w") as file:
    file.write(str(accuracy))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

np.savetxt("confusion_matrix.txt", cm, fmt="%d")

new_complaint = [
    "I am having trouble making my student loan payments and want to know about my repayment options."
]

new_prediction = pipeline.predict(new_complaint)

print("\nNew Complaint:")
print(new_complaint[0])

print("\nPredicted Product:")
print(new_prediction[0])

decision_scores = pipeline.decision_function(new_complaint)

print("\nDecision Scores:")
print(decision_scores)

print("\nClass Order:")
print(pipeline.classes_)

scores = decision_scores[0]

confidence = np.exp(scores) / np.exp(scores).sum()

predicted_class = np.argmax(scores)

print("\nConfidence Scores:")
for i in range(len(pipeline.classes_)):
    print(pipeline.classes_[i], ":", round(confidence[i] * 100, 2), "%")

print("\nPredicted Category Confidence:",
      round(confidence[predicted_class] * 100, 2), "%")