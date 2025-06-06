import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Clean and normalize text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+\s*Page[s]?\s*\d*', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# Segment resume into sections
def segment_cv(text):
    section_patterns = {
        "education": r"(education|academic background)",
        "experience": r"(experience|work history|employment)",
        "skills": r"(technical skills|skills)",
        "certifications": r"(certifications|courses)",
        "projects": r"(projects|research)",
        "achievements": r"(awards|achievements|honors)",
        "personal_info": r"(personal information|contact|profile)"
    }

    sections = defaultdict(str)
    lines = text.splitlines()
    current_section = None
    for line in lines:
        clean_line = line.strip().lower()
        for key, pattern in section_patterns.items():
            if re.search(pattern, clean_line):
                current_section = key
                break
        if current_section:
            sections[current_section] += line + "\n"

    return dict(sections)

#  Extract keywords using TF-IDF
def extract_keywords(text, top_n=30):
    cleaned_text = preprocess_text(text)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([cleaned_text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]

    top_n = min(top_n, len(feature_array))
    top_keywords = feature_array[tfidf_sorting][:top_n]

    return top_keywords.tolist()

# TF-IDF weights from JD
def assign_weights_from_jd(jd_text, top_n=30):
    cleaned_text = preprocess_text(jd_text)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([cleaned_text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_scores = tfidf_matrix.toarray().flatten()

    # Get indices sorted by tfidf score descending
    sorted_indices = tfidf_scores.argsort()[::-1]
    top_n = min(top_n, len(sorted_indices))
    top_indices = sorted_indices[:top_n]

    # Normalize weights between 1 and 3
    top_scores = tfidf_scores[top_indices]
    max_score = max(top_scores) if len(top_scores) > 0 else 1
    weighted_keywords = {
        feature_array[i]: round(1 + 2 * (top_scores[idx] / max_score), 2)
        for idx, i in enumerate(top_indices)
    }

    return weighted_keywords

#  Hardcoded constraint extraction from JD
def extract_constraints_from_jd(jd_title):
    constraints = {}

    if jd_title == "Research Assistant":
        constraints["skills"] = [
            "literature review",  "data collection","data analysis","interviewing","record keeping","confidentiality handling",
            "report writing","presentation preparation","academic writing","budget monitoring","research ethics","project coordination",
            "supervising undergraduates","equipment procurement","email communication","website content management", "progress reporting",
            "human subjects review","grant proposal preparation","experimental data handling"
        ]
    elif jd_title == "Lab Instructor":
        constraints["skills"] = [
            "lecture delivery", "tutorial conduction", "seminar facilitation", "undergraduate teaching", "postgraduate teaching", "distance learning","curriculum development",
            "module development", "course material preparation", "student assessment", "research publication", "academic writing", "interdepartmental collaboration",
            "course validation documentation", "marketing and outreach", "module leadership", "peer observation responsiveness","student feedback analysis",
            "external examiner coordination","liaison with professional bodies", "networking with schools and colleges","committee participation", "technical collaboration", "oral communication",
            "written communication", "learning material design", "high-quality teaching delivery","academic tutoring","team collaboration"
        ]
    return constraints

#  Extract skills from CV text
def extract_constraints(text):
    sections = segment_cv(text)
    skills_text = sections.get("skills", text)
    skills_text = preprocess_text(skills_text)
    tokens = skills_text.split()
    skills = list(set(tokens))
    return {
        "skills": skills
    }

# Entry points
def process_cv(text):
    cleaned_text = preprocess_text(text)
    keywords = extract_keywords(text)
    constraints = extract_constraints(text)
    return {
        "cleaned_text": cleaned_text,
        "keywords": keywords,
        "constraints": constraints
    }

def process_jd(text, job_title=None):
    cleaned_text = preprocess_text(text)
    keyword_weights = assign_weights_from_jd(text)

    if job_title in ["Research Assistant", "Lab Instructor"]:
        constraints = extract_constraints_from_jd( job_title)
    else:
        constraints = extract_constraints(text)
    return {
        "cleaned_text": cleaned_text,
        "keyword_weights": keyword_weights,
        "constraints": constraints,
        "title": job_title
    }
