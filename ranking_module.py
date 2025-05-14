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
# Constraint Filtering (optional)
# -------------------------------
def filter_by_constraints(cvs, jd_constraints):
    filtered = []
    for cv in cvs:
        cv_constraints = cv.get("constraints", {})
        degree_match = any(
            degree.lower() in [d.lower() for d in jd_constraints.get("degrees", [])]
            for degree in cv_constraints.get("degrees", [])
        )
        experience_match = (
            cv_constraints.get("years_experience", 0) >= jd_constraints.get("years_experience", 0)
        )

        if degree_match and experience_match:
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
        cvs = filter_by_constraints(cvs, jd["constraints"])
        if not cvs:
            return []

    # Cosine similarity (TF-IDF)
    cosine_scores = compute_similarity_scores(cvs, jd["cleaned_text"])

    # JD keyword weights (from JD preprocessing)
    jd_keyword_weights = jd.get("keyword_weights", {})

    ranked_results = []
    for i, cv in enumerate(cvs):
        kw_score = keyword_match_score(cv["keywords"], jd_keyword_weights)

        # Hybrid Score: weighted avg of cosine + normalized kw score
        combined_score = 0.7 * cosine_scores[i] + 0.3 * (kw_score / 100)

        cv["explanation"] = {
            "cosine_similarity": round(float(cosine_scores[i]), 4),
            "keyword_score": round(float(kw_score), 2),
            "final_score": round(float(combined_score), 4)
        }

        ranked_results.append((cv, combined_score))

    ranked_results.sort(key=lambda x: x[1], reverse=True)
    return ranked_results
