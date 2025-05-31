from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Cosine Similarity Using TF-IDF
# -------------------------------
def compute_similarity_scores(cvs, jd_text):
    documents = [cv['cleaned_text'] for cv in cvs] + [jd_text]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    jd_vector = tfidf_matrix[-1]
    cv_vectors = tfidf_matrix[:-1]

    scores = cosine_similarity(cv_vectors, jd_vector)
    return scores.flatten()


# -------------------------------
# Constraint Filtering (skills only)
# -------------------------------
def filter_by_constraints(cvs, jd_constraints):
    jd_skills = set(skill.lower() for skill in jd_constraints.get("skills", []))
    filtered = []
    for cv in cvs:
        cv_skills = set(skill.lower() for skill in cv.get("constraints", {}).get("skills", []))
        # Require at least one matching skill
        if jd_skills.intersection(cv_skills):
            filtered.append(cv)
    return filtered


# -------------------------------
# Weighted Keyword Score Matching
# -------------------------------
def keyword_match_score(cv_keywords, jd_keyword_weights):
    score = 0.0
    for word in cv_keywords:
        score += jd_keyword_weights.get(word, 0)
    return score


# -------------------------------
# Main Ranking Function
# -------------------------------
def rank_and_filter(cvs, jd, use_constraints=False):
    if use_constraints:
        cvs = filter_by_constraints(cvs, jd.get("constraints", {}))
        if not cvs:
            return []

    # Cosine similarity (TF-IDF)
    cosine_scores = compute_similarity_scores(cvs, jd["cleaned_text"])

    # JD keyword weights (from JD preprocessing)
    jd_keyword_weights = jd.get("keyword_weights", {})

    ranked_results = []
    for i, cv in enumerate(cvs):
        kw_score = keyword_match_score(cv.get("keywords", []), jd_keyword_weights)

        # Hybrid Score: weighted avg of cosine + normalized kw score
        combined_score = 0.7 * cosine_scores[i] + 0.3 * (kw_score / 100)
        ranked_results.append((cv, combined_score))

    ranked_results.sort(key=lambda x: x[1], reverse=True)
    return ranked_results
