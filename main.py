import streamlit as st
import time
import pandas as pd
from preprocessing import process_cv, process_jd                # shared preprocessing
from hybrid_module import train_naive_bayes, ensemble_predict  # Module C
from multilabel_module import initialize_models, classify       # Module B

# Map integer labels → role names for multi-label
INT_TO_ROLE = {
    0: "Lab Instructor",
    1: "Research Assistant",
    0: "Lab Engineer"
    
}

# ---------------------------------------------
# Helper to read uploaded files
# ---------------------------------------------
def read_file(file):
    return file.read().decode('latin-1')

# ---------------------------------------------
# Labeled data (filename → integer label)
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
st.markdown("Upload resumes to train and then classify/rank new ones for three roles.")
st.sidebar.title("Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Choose files", accept_multiple_files=True)

# ---------------------------------------------
# 1) Gather labeled CVs
# ---------------------------------------------
labeled_texts = []
labels        = []
for fname, lbl in label_data.items():
    f = next((x for x in uploaded_files if x.name == fname), None)
    if f:
        text = read_file(f)
        proc = process_cv(text)
        labeled_texts.append(proc["cleaned_text"])
        labels.append(lbl)

if not labeled_texts:
    st.warning("Please upload *all* labeled CV files to train the models.")
    st.stop()

# ---------------------------------------------
# 2) Train all three modules
# ---------------------------------------------
with st.spinner("Training Naive Bayes & Multi-Label models…"):
    time.sleep(1)
    # C: Naive Bayes for ensemble
    vec_nb, clf_nb, acc_nb = train_naive_bayes(labeled_texts, labels)
    # B: Multi-Label One-vs-Rest
    role_lists = [[INT_TO_ROLE[l]] for l in labels]
    clf_ml, vec_ml, mlb = initialize_models(labeled_texts, role_lists)

st.success(f"🔔 NB accuracy: {acc_nb*100:.2f}%; Multi-Label model ready")

# ---------------------------------------------
# 3) Load Job Descriptions
# ---------------------------------------------
jd_ra_f = next((x for x in uploaded_files if x.name == "JD Research Assistants.txt"), None)
jd_li_f = next((x for x in uploaded_files if x.name == "JD-Instructors.txt"), None)
if not jd_ra_f or not jd_li_f:
    st.error("Please upload both 'JD Research Assistants.txt' and 'JD-Instructors.txt'.")
    st.stop()

jd_ra = process_jd(read_file(jd_ra_f), job_title="Research Assistant")
jd_li = process_jd(read_file(jd_li_f), job_title="Lab Instructor")

# ---------------------------------------------
# 4) Classify & Rank New Resumes (Two Buttons)
# ---------------------------------------------

# Identify new CVs (not used for training and not JDs)
new_files = [
    x for x in uploaded_files
    if x.name not in label_data
    and x.name not in (jd_ra_f.name, jd_li_f.name)
]

if new_files:
    # Preprocess all new resumes once
    processed_new = []
    for f in new_files:
        d = process_cv(read_file(f))
        d["filename"] = f.name
        processed_new.append(d)

    st.markdown("## 🔎 Resume Analysis Options")
    col1, col2 = st.columns(2)

    with col1:
        show_ranking = st.button("📊 Show Ranked Resumes")
    with col2:
        show_labels = st.button("🏷️ Show Predicted Labels")

    if show_ranking:
        st.markdown("### 🔍 IR-Based Ranked Results")

        def show_ir(title, ranked):
            st.markdown(f"#### Top for {title}:")
            if not ranked:
                st.warning(f"No matches for {title}.")
            for i, (cv, _) in enumerate(ranked, 1):
                e = cv["explanation"]
                st.markdown(
                    f"**{i}. {cv['filename']}**  \n"
                    f"- IR Score: `{e['ir_score']:.4f}`  \n"
                    f"-  NB prob: `{e['nb_prob_ra']:.4f}`  \n"
                    f"- Combined Score: `{e['ensemble_score']:.4f}`"
                )

        show_ir("Research Assistant", ensemble_predict(processed_new, jd_ra, labeled_texts, labels))
        show_ir("Lab Instructor", ensemble_predict(processed_new, jd_li, labeled_texts, labels))

    if show_labels:
        st.markdown("### 🏷️ Multi-Label Predicted Roles")
        texts_new = [cv["cleaned_text"] for cv in processed_new]
        filenames_new = [cv["filename"] for cv in processed_new]
        df_ml = classify(texts_new, clf_ml, vec_ml, mlb, filenames=filenames_new)
        st.dataframe(df_ml.style.set_properties(**{'text-align': 'left'}))


else:
    st.info("No new resumes to classify.")
