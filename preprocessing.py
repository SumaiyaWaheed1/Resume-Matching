import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer

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
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


# -------------------------------
# Step 2: Segment resume into sections
# -------------------------------
def segment_cv(text):
    section_patterns = {
        "education": r"(education|academic background)",
        "experience": r"(experience|work history|employment|responsibilities|roles)",
        "skills": r"(technical skills|skills|skills & interests|IT skills|professional skills)",
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
# Step 3: Extract keywords from JD or CV
# -------------------------------
def extract_keywords(text, top_n=30):
    text = preprocess_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    freq = Counter(tokens)
    common = freq.most_common(top_n)
    return [word for word, count in common]


# -------------------------------
# Step 4: Assign dynamic weights (TF-IDF style)
# -------------------------------
def assign_weights_from_jd(jd_text, top_n=30):
    # Preprocess and extract keywords from JD text
    cleaned = preprocess_text(jd_text)
    tokens = word_tokenize(cleaned)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    freq = Counter(tokens)
    top_keywords = dict(freq.most_common(top_n))

    # Normalize weights between 1 and 3
    max_count = max(top_keywords.values(), default=1)
    weighted_keywords = {kw: round(1 + 2 * (count / max_count), 2) for kw, count in top_keywords.items()}

    return weighted_keywords


# -------------------------------
# Step 5: Constraint Extraction
# -------------------------------
def extract_constraints(text):
    degree_patterns = [
        r"bachelor(?:s)?(?: of [a-z\s]+)?", r"bs[c]?", r"m[\.]?\s?sc",
        r"ph\.?d", r"m\.?phil", r"master(?:s)?(?: of [a-z\s]+)?"
    ]
    degrees = []
    for pattern in degree_patterns:
        degrees += re.findall(pattern, text.lower())

    years = re.findall(r"(\d+)\+?\s+(?:years|yrs)", text.lower())
    max_years = max([int(y) for y in years], default=0)

    return {
        "degrees": list(set(degrees)),
        "years_experience": max_years
    }


# -------------------------------
# Final Entry Points
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