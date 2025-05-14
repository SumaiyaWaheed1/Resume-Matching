import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

from ranking_module import rank_and_filter  # 🆕 ADDED
from preprocessing import process_cv, process_jd

# ---------------------------------------------
# Helper function to read uploaded files
# ---------------------------------------------
def read_file(file):
    return file.read().decode('latin-1')

# ---------------------------------------------
# Labeled data
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
st.title("Resume Classification & Ranking App")
st.markdown("Upload resumes to classify and rank them for Research Assistant or Lab Instructor roles.")

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
        labeled_resumes.append(cv_data["cleaned_text"])  # 🔁 MODIFIED
        labels.append(label)

if labeled_resumes:
    with st.spinner('Processing and training the model...'):
        time.sleep(2)

    # ---------------------------------------------
    # Job Descriptions
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
        # Predict and Rank New Resumes
        # ---------------------------------------------
        if st.sidebar.button("Classify New Resumes"):
            unlabeled_files = [
                f for f in uploaded_files
                if f.name not in label_data and f.name not in ["JD Research Assistants.txt", "JD-Instructors.txt"]
            ]

            if not unlabeled_files:
                st.info("No new resumes to classify.")
            else:
                processed_unlabeled = []
                for file in unlabeled_files:
                    cv_data = process_cv(read_file(file))
                    cv_data["filename"] = file.name  # 🆕 Keep filename for display
                    processed_unlabeled.append(cv_data)

                # ---------------------------------------------
                # Use ranking module
                # ---------------------------------------------
                st.markdown("### 🔍 Ranked Results (Cosine Similarity + Constraints + Keyword Weighting)")

                def display_ranked_results(title, ranked_list):
                    st.markdown(f"#### 📌 Top Candidates for {title}:")
                    if ranked_list:
                        for rank, (cv, score) in enumerate(ranked_list, start=1):
                            st.markdown(f"""
                            **{rank}. {cv['filename']}**
                            - 🔢 Final Score: `{cv['explanation']['final_score']:.4f}`
                            - 📊 Cosine Similarity: `{cv['explanation']['cosine_similarity']:.4f}`
                            - 🧠 Keyword Score: `{cv['explanation']['keyword_score']}/100`
                            """)
                    else:
                        st.warning(f"No matching CVs for {title} after filtering.")

                # Display ranked candidates for each job
                display_ranked_results("Research Assistant", rank_and_filter(processed_unlabeled, jd_ra_data))
                display_ranked_results("Lab Instructor", rank_and_filter(processed_unlabeled, jd_li_data))

                # ---------------------------------------------
                # Optional: Classification-based prediction
                # ---------------------------------------------
                unlabeled_cleaned = [cv["cleaned_text"] for cv in processed_unlabeled]
                X_unlabeled = vectorizer.transform(unlabeled_cleaned)
                unlabeled_preds = clf.predict(X_unlabeled)

                st.markdown("### 🧪 Naive Bayes Predictions")
                for cv, prediction in zip(processed_unlabeled, unlabeled_preds):
                    label = "Research Assistant" if prediction == 1 else "Lab Instructor"
                    st.write(f"📄 **{cv['filename']}** → 🏷️ Predicted Label: **{label}**")

    else:
        st.error("Please upload both job description files: 'JD Research Assistants.txt' and 'JD-Instructors.txt'.")

else:
    st.warning("Please upload labeled resumes to train the model.")
