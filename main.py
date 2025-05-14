import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk import download
import time

from preprocessing import process_cv, process_jd

# ---------------------------------------------
# Helper function to read uploaded files
# ---------------------------------------------
def read_file(file):
    return file.read().decode('latin-1')

# ---------------------------------------------
# Labeled data: filenames and their labels
# ---------------------------------------------
label_data = {
    "Umair Ahmed - CV for Lab Instructor.txt": 0,
    "Hafiz Ali Raja - CV for Lab Instructor.txt": 0,
    "Faisal Shahzad - CV for Lab Instructor.txt": 0,
    "Saad Hassan Khan - CV for Research Assistant (1).txt": 1,
    "Waqas Ahmed - CV for Lab Instructor.txt": 0,
    "Muhammad Omaid Sheikh - CV for for Lab Instructor.txt": 0,
    "Awais Anwar - CV for Lab Instructor.txt": 0,
    "Urooj Sheikh - CV for Lab Engineer.txt": 0,
    "Ghulam Jaffar - CV for Research Assistant.txt": 1,
    "Ebad Ali - CV for Research Assistant.txt": 1,
    "Sana Fatima - CV for Research Assistant.txt": 1,
    "Muhammad Tayyab Yaqoob - CV for Lab Engineer.txt": 0,
    "Haris Ahmed - CV for Research Assistant.txt": 1,
    "Faisal Nisar - CV for for Research Assistant.txt": 1,
    "Muhammad Azmi Umer - CV for Lab Instructor.txt": 0,
    "Shawana Khan - CV for Lab Instructor (1).txt": 0
}

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
st.title("Resume Classification App")
st.markdown("Upload resumes to classify them as either 'Lab Instructor' or 'Research Assistant'.")

st.sidebar.title("Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Choose files", accept_multiple_files=True)

# ---------------------------------------------
# Process labeled resumes
# ---------------------------------------------
labeled_resumes = []
labels = []

for filename, label in label_data.items():
    file = next((f for f in uploaded_files if f.name == filename), None)
    if file is not None:
        file_content = read_file(file)
        cv_data = process_cv(file_content)
        labeled_resumes.append(cv_data["cleaned_text"])  # TF-IDF input
        labels.append(label)

if labeled_resumes:
    with st.spinner('Processing and training the model...'):
        time.sleep(2)

    # ---------------------------------------------
    # Process Job Descriptions
    # ---------------------------------------------
    jd_ra_file = next((f for f in uploaded_files if f.name == "JD Research Assistants.txt"), None)
    jd_li_file = next((f for f in uploaded_files if f.name == "JD-Instructors.txt"), None)

    if jd_ra_file and jd_li_file:
        jd_ra_data = process_jd(read_file(jd_ra_file), job_title="Research Assistant")
        jd_li_data = process_jd(read_file(jd_li_file), job_title="Lab Instructor")

        # ---------------------------------------------
        # Train TF-IDF + Naive Bayes classifier
        # ---------------------------------------------
        vectorizer = TfidfVectorizer()
        X_labeled = vectorizer.fit_transform(labeled_resumes)

        X_train, X_test, y_train, y_test = train_test_split(X_labeled, labels, test_size=0.2, random_state=48)
        clf = MultinomialNB()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Model trained with accuracy: {accuracy * 100:.2f}%")

        # ---------------------------------------------
        # Predict new (unlabeled) resumes
        # ---------------------------------------------
        if st.sidebar.button("Classify New Resumes"):
            unlabeled_files = [
                f for f in uploaded_files
                if f.name not in label_data and f.name not in ["JD Research Assistants.txt", "JD-Instructors.txt"]
            ]

            if not unlabeled_files:
                st.info("No new resumes to classify.")
            else:
                unlabeled_resumes = [process_cv(read_file(file))["cleaned_text"] for file in unlabeled_files]
                X_unlabeled = vectorizer.transform(unlabeled_resumes)
                unlabeled_preds = clf.predict(X_unlabeled)

                st.markdown("### Prediction Results")
                for file, prediction in zip(unlabeled_files, unlabeled_preds):
                    label = "Research Assistant" if prediction == 1 else "Lab Instructor"
                    st.write(f"📄 **{file.name}** → 🏷️ *{label}*")

    else:
        st.error("Please upload both job description files: 'JD Research Assistants.txt' and 'JD-Instructors.txt'.")

else:
    st.warning("Please upload labeled resumes to train the model.")
