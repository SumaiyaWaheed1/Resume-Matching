import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# 🔽 Download required NLTK resources
# -------------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------------
# Step 1: Clean and normalize text
# -------------------------------
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+\s*Page[s]?\s*\d*', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# -------------------------------
# Step 2: Segment resume into sections
# -------------------------------
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

# -------------------------------
# Step 3: Extract keywords using TF-IDF
# -------------------------------
def extract_keywords(text, top_n=30):
    cleaned_text = preprocess_text(text)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([cleaned_text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]

    top_n = min(top_n, len(feature_array))
    top_keywords = feature_array[tfidf_sorting][:top_n]

    return top_keywords.tolist()

# -------------------------------
# Step 4: Assign TF-IDF weights from JD
# -------------------------------
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

# -------------------------------
# Step 5: Extract skills as constraints from Skills section or whole text
# -------------------------------
def extract_constraints(text):
    # Try to find skills section first
    sections = segment_cv(text)
    skills_text = sections.get("skills", text)

    # Preprocess skills text
    skills_text = preprocess_text(skills_text)
    tokens = skills_text.split()

    # Simple heuristic: extract nouns and noun phrases as skills (optional: can be enhanced)
    # For now, just return unique tokens as skills
    skills = list(set(tokens))

    return {
        "skills": skills
    }

# -------------------------------
# Entry points
# -------------------------------
def process_cv(text):
    cleaned_text = preprocess_text(text)
    segmented = segment_cv(text)
    keywords = extract_keywords(text)
    constraints = extract_constraints(text)
    return {
        "cleaned_text": cleaned_text,
        "segmented": segmented,
        "keywords": keywords,
        "constraints": constraints
    }

def process_jd(text, job_title=None):
    cleaned_text = preprocess_text(text)
    segmented = segment_cv(text)
    keyword_weights = assign_weights_from_jd(text)
    constraints = extract_constraints(text)
    return {
        "cleaned_text": cleaned_text,
        "segmented": segmented,
        "keyword_weights": keyword_weights,
        "constraints": constraints,
        "job_title": job_title
    }
