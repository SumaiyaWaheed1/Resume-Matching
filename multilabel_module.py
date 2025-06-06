import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from preprocessing import process_cv
import pandas as pd
from collections import Counter
from ranking_module import compute_similarity_scores

# 1) Initialize spaCy and add EntityRuler (for SKILL patterns, etc.)
nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")

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

# 2) Define your two valid roles (Lab Instructor, Research Assistant)
ROLES = ["Lab Instructor", "Research Assistant"]

# 3) Only keep these entity types when extracting NER; SKILL comes from rulings
VALID_LABELS = {"SKILL", "ORG", "PRODUCT", "LANGUAGE", "GPE"}

def extract_ner_from_processed(processed):
   
    #running spacy on cleaned text and return list of labels
    text = processed["cleaned_text"]
    doc = nlp(text)
    seen = set()
    entities = []
    for ent in doc.ents:
        label = ent.label_
        tok = ent.text.strip()
        tok_norm = tok.lower().strip(".,;:\"'()[]")
        # Skip if not in VALID_LABELS, empty after normalization, or contains any digit
        if label not in VALID_LABELS or not tok_norm or any(ch.isdigit() for ch in tok_norm):
            continue
        key = (tok_norm, label)
        if key not in seen:
            seen.add(key)
            entities.append(key)
    return entities


def initialize_models(texts, labels):
    """
    1) Map any "Lab Engineer" labels → "Lab Instructor".
    2) (Optional) If labels are encoded as integers, map them to ROLES.
    3) Print out raw label distribution (post-mapping).
    4) Print NER entities for each training CV (debug).
    5) Vectorize cleaned_text via TF-IDF (unigrams + bigrams, min_df=2, stop words).
    6) Binarize multi-labels using MultiLabelBinarizer(classes=ROLES).
    7) Train One-vs-Rest Logistic Regression with class_weight="balanced".
    8) Diagnostic: confirm sub-estimators and class weights.
    9) Attach the trained vectorizer to clf and return (clf, vect, mlb).
    """
    # Replace "Lab Engineer" with "Lab Instructor" in each label‐list
    mapped_labels = []
    for lab_list in labels:
        new_list = []
        for lbl in lab_list:
            if lbl == "Lab Engineer":
                new_list.append("Lab Instructor")
            else:
                new_list.append(lbl)
        # Remove duplicates if both were present
        new_list = list(dict.fromkeys(new_list))
        mapped_labels.append(new_list)
    labels = mapped_labels

    if labels and isinstance(labels[0], list) and labels[0] and isinstance(labels[0][0], int):
        labels = [[ROLES[idx] for idx in lab_list] for lab_list in labels]



    
    print("=== NER Entities in Training Set ===")
    for i, text in enumerate(texts):
        proc = process_cv(text)
        ents = extract_ner_from_processed(proc)
        print(f"Doc {i+1} Entities:", ents)
    print("=== End NER Entities ===\n")

    # 5) TF-IDF vectorization on cleaned text
    cleaned = [process_cv(doc)["cleaned_text"] for doc in texts]
    vect = TfidfVectorizer(
        max_features=3000,        # keep top 3000 features
        ngram_range=(1, 3),       # include unigrams + bigrams
        min_df=2,                 # token must appear in ≥2 documents
        stop_words="english"      # drop common English stop words
    )
    X = vect.fit_transform(cleaned)

    #  Binarize labels into a 2-column matrix (LI vs. RA)
    mlb = MultiLabelBinarizer(classes=ROLES)
    Y = mlb.fit_transform(labels)

    # Train One-vs-Rest Logistic Regression with balanced class weights
    clf = OneVsRestClassifier(
        LogisticRegression(
            solver="saga",
            penalty="l2",
            class_weight="balanced",
            C=1.0,
            max_iter=2000
        )
    )
    clf.fit(X, Y)


    clf._vectorizer = vect
    return clf, vect, mlb


def classify(texts, clf, vect, mlb, filenames=None, jd_ra=None, jd_li=None):
   


    
    X_new = vect.transform(texts)

  
    probs = clf.predict_proba(X_new)
    

    # 4) Probability thresholds
    LI_prob_thresh = 0.45
    RA_prob_thresh = 0.45

  
    cvs_dicts   = [{"cleaned_text": t} for t in texts]
    sim_li_list = compute_similarity_scores(cvs_dicts, jd_li["cleaned_text"])
    sim_ra_list = compute_similarity_scores(cvs_dicts, jd_ra["cleaned_text"]) 

   
    LOW_THRESH = 0.05
    DELTA      = 0.03

    labels_pred = []
    for i, txt in enumerate(texts):
        p_li, p_ra = probs[i]
        chosen = []

        #  Use logistic 
        if p_li >= LI_prob_thresh and p_li >= p_ra:
            chosen = ["Lab Instructor"]
        elif p_ra >= RA_prob_thresh and p_ra >= p_li:
            chosen = ["Research Assistant"]
        else:
            # Fallback to cosine if neither prob is strong enough
            sim_li = sim_li_list[i]
            sim_ra = sim_ra_list[i]

            

            if sim_li < LOW_THRESH and sim_ra < LOW_THRESH:
                chosen = []
            elif sim_li >= LOW_THRESH and sim_ra >= LOW_THRESH and abs(sim_li - sim_ra) <= DELTA:
                chosen = ["Lab Instructor", "Research Assistant"]
            elif sim_li > sim_ra and sim_li >= LOW_THRESH:
                chosen = ["Lab Instructor"]
            elif sim_ra > sim_li and sim_ra >= LOW_THRESH:
                chosen = ["Research Assistant"]
            else:
                chosen = []

        
        cleaned_lower = txt.lower()
        if "lab instructor" in cleaned_lower and chosen != ["Lab Instructor", "Research Assistant"]:
            chosen = ["Lab Instructor"]
        if "research assistant" in cleaned_lower and chosen != ["Lab Instructor", "Research Assistant"]:
            chosen = ["Research Assistant"]

        labels_pred.append(chosen)

    #  Build the output DataFrame
    if filenames is None:
        filenames = [f"CV_{i+1}" for i in range(len(texts))]

    df = pd.DataFrame({
        "File name":       filenames,
        "Predicted Role": [", ".join(lbls) if lbls else "None" for lbls in labels_pred]
    })
    return df