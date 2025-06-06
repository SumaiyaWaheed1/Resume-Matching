import streamlit as st
import time
from preprocessing import process_cv, process_jd
from hybrid_module import train_naive_bayes, ensemble_predict
from multilabel_module import initialize_models, classify

# Label mapping
INT_TO_ROLE = {
    0: "Lab Instructor",   # Also used for Lab Engineer
    1: "Research Assistant"
}

# Labeled dataset mapping
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

# File reading helper
def read_file(file):
    return file.read().decode('latin-1')

# App Header
st.title("Resume Classification and Ranking System")
st.markdown("Upload CVs and JDs to classify and rank candidates for academic job roles.")

# Sidebar Upload
st.sidebar.title("Upload Files")
uploaded_files = st.sidebar.file_uploader("Upload CVs and JDs", accept_multiple_files=True)

# Load labeled CVs
labeled_texts, labels = [], []
for fname, lbl in label_data.items():
    file = next((f for f in uploaded_files if f.name == fname), None)
    if file:
        content = read_file(file)
        cleaned = process_cv(content)
        labeled_texts.append(cleaned["cleaned_text"])
        labels.append(lbl)

if not labeled_texts:
    st.warning("Please upload all required labeled CV files.")
    st.stop()

# Train models
with st.spinner("Training models on labeled CVs..."):
    time.sleep(1)
    vec_nb, clf_nb, acc_nb = train_naive_bayes(labeled_texts, labels)
    role_lists = [[INT_TO_ROLE[l]] for l in labels]
    clf_ml, vec_ml, mlb = initialize_models(labeled_texts, role_lists)
st.success("Models trained successfully.")

# Load and check job descriptions
st.markdown("---")
st.subheader("Step 2: Upload Job Descriptions")

jd_ra_file = next((f for f in uploaded_files if f.name == "JD Research Assistants.txt"), None)
jd_li_file = next((f for f in uploaded_files if f.name == "JD-Instructors.txt"), None)

if not jd_ra_file or not jd_li_file:
    st.error("Please upload both 'JD Research Assistants.txt' and 'JD-Instructors.txt'.")
    st.stop()

# Process job descriptions
jd_ra = process_jd(read_file(jd_ra_file), job_title= "Research Assistant")
jd_li = process_jd(read_file(jd_li_file), job_title= "Lab Instructor")
st.success("Job Descriptions processed successfully.")

# Upload new resumes
st.markdown("---")
st.subheader("Step 3: Upload New Resumes to Analyze")

new_resumes = [
    f for f in uploaded_files
    if f.name not in label_data and f.name not in (jd_ra_file.name, jd_li_file.name)
]

if not new_resumes:
    st.info("Please upload new resumes for classification or ranking.")
    st.stop()

# Process new resumes
processed_new = []
for file in new_resumes:
    data = process_cv(read_file(file))
    data["filename"] = file.name
    processed_new.append(data)

# Analysis Options
st.markdown("---")
st.subheader("Step 4: Choose Analysis Options")

col1, col2 = st.columns(2)
with col1:
    show_ranking = st.button("Show Ranked Resumes")
with col2:
    show_labels = st.button("Show Predicted Labels")

# Ranking display
if show_ranking:
    st.markdown("---")
    st.header("Resume Ranking Based on JD Similarity and NB Classification")

    ranked_ra = ensemble_predict(processed_new, jd_ra, labeled_texts, labels)
    ranked_li = ensemble_predict(processed_new, jd_li, labeled_texts, labels)

    def display_ranking(title, ranked_list):
        st.markdown(f"### Top Candidates for **{title}**")
        if not ranked_list:
            st.info("No matching resumes found.")
            return

        for idx, (cv, _) in enumerate(ranked_list, 1):
            explanation = cv.get("explanation_per_jd", {}).get(title, {})
            filename = cv.get("filename", "Unnamed Resume")

            with st.expander(f"{idx}. {filename}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Cosine Similarity", f"{explanation.get('ir_score', 0):.4f}")
                col2.metric("Naive Bayes", f"{explanation.get('nb_prob', 0):.4f}")
                col3.metric("Combined Score", f"{explanation.get('ensemble_score', 0):.4f}")

    display_ranking("Research Assistant", ranked_ra)
    st.markdown("---")
    display_ranking("Lab Instructor", ranked_li)

# Label classification display
if show_labels:
    st.markdown("---")
    st.header("Predicted Job Role(s) for New Resumes")

    texts_new = [cv["cleaned_text"] for cv in processed_new]
    filenames_new = [cv["filename"] for cv in processed_new]

    df_ml = classify(
        texts_new,
        clf_ml,
        vec_ml,
        mlb,
        filenames=filenames_new,
        jd_ra=jd_ra,
        jd_li=jd_li
    )

    st.dataframe(df_ml.style.set_properties(**{'text-align': 'left'}))
