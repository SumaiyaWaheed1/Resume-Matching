import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------------------------------
# Step 1: Clean and normalize text
# -------------------------------------------------
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+\s*Page[s]?\s*\d*', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(tokens)

# -------------------------------------------------
# Step 2: Segment resume into sections
# -------------------------------------------------
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

# -------------------------------------------------
# Step 3: Extract keywords from segmented sections
# -------------------------------------------------
def extract_keywords_from_sections(segmented_cv):
    tech_keywords = ["python", "java", "c#", "sql", "html", "css", "react", "node", "matlab"]
    tools_keywords = ["visual studio", "jupyter", "git", "jira", "github"]
    research_keywords = [
        "machine learning", "deep learning", "ai", "data science", "nlp", 
        "literature review", "data analysis", "research", "grant", 
        "summarize", "supervise", "project", "report", "submission"
    ]
    teaching_keywords = [
        "teaching", "curriculum", "lecture", "assessment", "academic", 
        "course material", "personal tutor", "distance learning", "student support"
    ]
    role_keywords = ["instructor", "assistant", "associate", "researcher", "tutor"]

    extracted = {"technologies": set(), "tools": set(), "research": set(), "roles": set(), "teaching": set()}
    for section, content in segmented_cv.items():
        text = content.lower()
        for word in tech_keywords:
            if word in text:
                extracted["technologies"].add(word)
        for tool in tools_keywords:
            if tool in text:
                extracted["tools"].add(tool)
        for keyword in research_keywords:
            if keyword in text:
                extracted["research"].add(keyword)
        for keyword in teaching_keywords:
            if keyword in text:
                extracted["teaching"].add(keyword)
        for role in role_keywords:
            if role in text:
                extracted["roles"].add(role)
    return {k: list(v) for k, v in extracted.items()}

# -------------------------------------------------
# Step 4: Constraint Extraction (degree, experience)
# -------------------------------------------------
def extract_constraints(text):
    degrees = re.findall(r"(bachelors|bs|msc|ms|phd|bsc|m\.phil)", text.lower())
    years = re.findall(r"(\d+)\+?\s+(?:years|yrs)", text.lower())

    max_years = max([int(y) for y in years], default=0)

    return {
        "degrees": list(set(degrees)),
        "years_experience": max_years
    }

# -------------------------------------------------
# Step 5: Assign job-specific weights (optional scoring module)
# -------------------------------------------------
def assign_weights(extracted_keywords, job_title):
    JD_KEYWORDS = {
        "Lab Instructor": {
            "instructor": 3, "teaching": 2.5, "curriculum": 2, "module": 1.5, "lecture": 2,
            "tutor": 2, "assessment": 1.5, "academic": 2, "course material": 1.5,
            "personal tutor": 2, "collaborate": 1, "research": 1, "feedback": 1.5,
            "university": 1, "cs": 1.5, "student support": 2, "distance learning": 1,
            "peer review": 1, "project proposal": 1
        },
        "Research Assistant": {
            "research": 3, "literature review": 2.5, "data analysis": 2.5, "machine learning": 2,
            "python": 2, "matlab": 1.5, "ai": 2, "nlp": 1.5, "grant": 1.5,
            "submission": 1, "interview": 1.5, "summarize": 1, "report": 1.5,
            "project": 2, "budget": 1, "pi": 1.2, "presentation": 1.5,
            "supervise": 1.5, "progress report": 1, "analyze data": 2
        }
    }

    job_weights = JD_KEYWORDS.get(job_title, {})
    score = 0
    for category in extracted_keywords:
        for kw in extracted_keywords[category]:
            score += job_weights.get(kw, 0)

    return score

# -------------------------------------------------
# Final Entry Point: Process a single CV
# -------------------------------------------------
def process_cv(text):
    cleaned_text = preprocess_text(text)
    segmented = segment_cv(text)
    extracted_keywords = extract_keywords_from_sections(segmented)
    constraints = extract_constraints(text)
    return {
        "cleaned_text": cleaned_text,      # For TF-IDF
        "segmented": segmented,            # For deep parsing if needed
        "keywords": extracted_keywords,    # For role/label/ranking
        "constraints": constraints         # For filtering in IR ranking
    }

def process_jd(text, job_title=None):
    cleaned_text = preprocess_text(text)
    segmented = segment_cv(text)
    extracted_keywords = extract_keywords_from_sections(segmented)
    constraints = extract_constraints(text)

    # Use the title if provided to add roles
    if job_title:
        extracted_keywords["roles"].append(job_title.lower())

    return {
        "cleaned_text": cleaned_text,
        "segmented": segmented,
        "keywords": extracted_keywords,
        "constraints": constraints,
        "job_title": job_title
    }
