import streamlit as st
import time

from ranking_module import rank_and_filter
from preprocessing import process_cv, process_jd
from hybrid_module import (
    train_naive_bayes,
    predict_naive_bayes,
    ensemble_predict
)

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
# Gather labeled resumes
# ---------------------------------------------
labeled_resumes = []
labels = []

for filename, label in label_data.items():
    file = next((f for f in uploaded_files if f.name == filename), None)
    if file:
        content = read_file(file)
        cv_data = process_cv(content)
        labeled_resumes.append(cv_data["cleaned_text"])
        labels.append(label)

if labeled_resumes:
    with st.spinner("Processing and training the model..."):
        time.sleep(2)

    # Train NB
    vectorizer, clf, accuracy = train_naive_bayes(labeled_resumes, labels)
    st.success(f"Model trained with accuracy: {accuracy * 100:.2f}%")

    # Load JDs
    jd_ra_file = next((f for f in uploaded_files if f.name == "JD Research Assistants.txt"), None)
    jd_li_file = next((f for f in uploaded_files if f.name == "JD-Instructors.txt"), None)

    if jd_ra_file and jd_li_file:
        jd_ra_data = process_jd(read_file(jd_ra_file), job_title="Research Assistant")
        jd_li_data = process_jd(read_file(jd_li_file), job_title="Lab Instructor")

        if st.sidebar.button("Classify New Resumes"):
            unlabeled = [
                f for f in uploaded_files
                if f.name not in label_data
                and f.name not in ("JD Research Assistants.txt", "JD-Instructors.txt")
            ]

            if not unlabeled:
                st.info("No new resumes to classify.")
            else:
                processed_unlabeled = []
                for file in unlabeled:
                    cv = process_cv(read_file(file))
                    cv["filename"] = file.name
                    processed_unlabeled.append(cv)

                # 1) IR rankings (unchanged)
                st.markdown("### 🔍 Ranked Results (Cosine Similarity + Constraints + Keyword Weighting)")
                def display_ranked(title, ranked):
                    st.markdown(f"#### 📌 Top Candidates for {title}:")
                    if ranked:
                        for i, (cv, _) in enumerate(ranked, 1):
                            exp = cv["explanation"]
                            st.markdown(f"""
**{i}. {cv['filename']}**  
- 🔢 Final Score: `{exp['final_score']:.4f}`  
- 📊 Cosine Similarity: `{exp['cosine_similarity']:.4f}`  
- 🧠 Keyword Score: `{exp['keyword_score']}/100`  
""")
                    else:
                        st.warning(f"No matching CVs for {title} after filtering.")

                display_ranked("Research Assistant", rank_and_filter(processed_unlabeled, jd_ra_data))
                display_ranked("Lab Instructor",      rank_and_filter(processed_unlabeled, jd_li_data))

                # 2) Ensemble Predictions (using hybrid_module defaults alpha=0.4, threshold=0.45)
                st.markdown("### 🤝 Ensemble Predictions (IR + NB)")
                ensemble = ensemble_predict(
                    processed_unlabeled,
                    jd_ra_data,
                    labeled_resumes,
                    labels
                )
                for cv, label in ensemble:
                    exp = cv["explanation"]
                    st.markdown(f"""
**{cv['filename']}**  
- IR Score: `{exp['ir_score']}`  
- NB P(RA): `{exp['nb_prob_ra']}`  
- Ensemble Score: `{exp['ensemble_score']}`  
→ **{label}**
""")
    else:
        st.error("Please upload both job description files: 'JD Research Assistants.txt' and 'JD-Instructors.txt'.")
else:
    st.warning("Please upload labeled resumes to train the model.")
