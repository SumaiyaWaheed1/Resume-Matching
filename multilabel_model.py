import re
import spacy
from spacy.pipeline import EntityRuler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
#from sklearn.linear_model import LogisticRegression  # no longer needed
from preprocessing import process_cv, process_jd ,extract_keywords
import pandas as pd
import glob
from collections import Counter

# Initialize spaCy and add EntityRuler
nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Fallback: default skill patterns
SKILL_PATTERNS = [
    {"label": "SKILL", "pattern": "python"},
    {"label": "SKILL", "pattern": "java"},
    {"label": "SKILL", "pattern": "c++"},
    {"label": "SKILL", "pattern": "c#"},
    {"label": "SKILL", "pattern": "javascript"},
    {"label": "SKILL", "pattern": "matlab"},
    {"label": "SKILL", "pattern": "sql"},
    {"label": "SKILL", "pattern": "tensorflow"},
    {"label": "SKILL", "pattern": "react"},
    {"label": "SKILL", "pattern": "node"},
    {"label": "SKILL", "pattern": "django"},
    {"label": "SKILL", "pattern": "flask"},
    {"label": "SKILL", "pattern": "hadoop"},
    {"label": "SKILL", "pattern": "docker"},
    {"label": "SKILL", "pattern": "kubernetes"},
]
ruler.add_patterns(SKILL_PATTERNS)

# Your three roles
ROLES = ["Lab Instructor", "Research Assistant", "Lab Engineer"]

# Which entity labels to keep
VALID_LABELS = {"SKILL", "ORG", "PRODUCT", "LANGUAGE", "GPE"}


def extract_ner_from_processed(processed):
    """
    Extract entities from cleaned_text, including SKILLs from the rule-based ruler.
    """
    text = processed["cleaned_text"]
    doc = nlp(text)
    seen = set()
    entities = []
    for ent in doc.ents:
        label = ent.label_
        tok = ent.text.strip()
        # normalize
        tok_norm = tok.lower().strip(".,;:\"'()[]")
        if label not in VALID_LABELS or not tok_norm or any(ch.isdigit() for ch in tok_norm):
            continue
        key = (tok_norm, label)
        if key not in seen:
            seen.add(key)
            entities.append(key)
    return entities


def initialize_models(texts, labels):
    # Handle numeric labels by mapping to role names
    if labels and isinstance(labels[0], list) and labels[0] and isinstance(labels[0][0], int):
        labels = [[ROLES[lbl] for lbl in label_list] for label_list in labels]
    # 1) Demonstrate NER on processed training set
    # 1) Demonstrate NER on processed training set
    print("=== NER Entities in Training Set ===")
    for i, text in enumerate(texts):
        processed = process_cv(text)
        ents = extract_ner_from_processed(processed)
        print(f"Doc {i+1} Entities:", ents)
    print("=== End NER Entities ===\n")

    # 2) Preprocess and vectorize cleaned text
    cleaned = [ process_cv(doc)["cleaned_text"] for doc in texts ]
    vect   = TfidfVectorizer(max_features=5000)
    X      = vect.fit_transform(cleaned)

    # 3) Binarize labels
    mlb = MultiLabelBinarizer(classes=ROLES)
    Y   = mlb.fit_transform(labels)

    # 4) Train classifier using MultinomialNB in One-vs-Rest
    clf = OneVsRestClassifier(
        MultinomialNB()
    )
    clf.fit(X, Y)
    # attach trained vectorizer for prediction consistency
    clf._vectorizer = vect

    return clf, vect, mlb


def classify(texts, clf, vect, mlb):
    # 1) Demonstrate NER on processed prediction set
    print("=== NER Entities in Prediction Set ===")
    for i, text in enumerate(texts):
        processed = process_cv(text)
        ents = extract_ner_from_processed(processed)
        print(f"Unlabeled Doc {i+1} Entities:", ents)
    print("=== End NER Entities ===")

    # 2) Preprocess & vectorize cleaned text with the original training vectorizer
    cleaned = [ process_cv(doc)["cleaned_text"] for doc in texts ]
    vect_to_use = getattr(clf, '_vectorizer', vect)
    X_new   = vect_to_use.transform(cleaned)
    cleaned = [ process_cv(doc)["cleaned_text"] for doc in texts ]
    X_new   = vect.transform(cleaned)

    # 3) Predict with probability threshold for multi-label
    # get probability estimates for each class
    probs = clf.predict_proba(X_new)
    labels_pred = []
    for sample_probs in probs:
        # pick all roles above threshold
        threshold = 0.3
        selected = [mlb.classes_[i] for i, p in enumerate(sample_probs) if p >= threshold]
        # if none meet threshold, fall back to the highest-prob role
        if not selected:
            max_idx = sample_probs.argmax()
            selected = [mlb.classes_[max_idx]]
        labels_pred.append(selected)

    # 4) Build DataFrame) Build DataFrame
    df = pd.DataFrame({
        "filename": [f"CV_{i+1}" for i in range(len(texts))],
        "predicted_roles": [", ".join(lbls) if lbls else "None" for lbls in labels_pred]
    })
    return df

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python multilabel_model.py <path_to_text_file>")
        sys.exit(1)
    text = open(sys.argv[1], encoding='utf-8').read()
    processed = process_cv(text)
    ents = extract_ner_from_processed(processed)
    for ent, label in ents:
        print(f"{ent} ({label})")
