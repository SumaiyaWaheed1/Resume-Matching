import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from scipy import sparse
from ranking_module import rank_and_filter


def train_naive_bayes(labeled_texts, labels, test_size=0.2, random_state=48):
    """
    Train a TF-IDF + Naive Bayes classifier on labeled texts with oversampling,
    tuned TF-IDF parameters to mitigate overfitting, and return 5-fold CV accuracy.
    """
    # Build a pipeline for cross-validation
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 1),    # unigrams only
            min_df=2,
            max_df=0.7,
            max_features=1000
        )),
        ('nb',   MultinomialNB(alpha=1.5))
    ])
    # 1) Compute 5-fold cross-validation accuracy on the full labeled set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline,labeled_texts,labels, cv=cv,scoring='accuracy',n_jobs=-1)
    cv_accuracy = cv_scores.mean()

    # 2) Now fit the vectorizer & classifier exactly as before (for downstream use)
    vectorizer = pipeline.named_steps['tfidf']
    clf       = pipeline.named_steps['nb']

    # Vectorize all labeled_texts
    X = vectorizer.fit_transform(labeled_texts)

    # Stratified hold-out split (still performed, but accuracy not returned)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # Oversample the minority class in the TRAIN set only
    y_train = np.array(y_train)
    X_train = sparse.csr_matrix(X_train)
    ra_mask = (y_train == 1)
    li_mask = (y_train == 0)
    X_ra, y_ra = X_train[ra_mask], y_train[ra_mask]
    X_li, y_li = X_train[li_mask], y_train[li_mask]

    X_ra_up, y_ra_up = resample(
        X_ra, y_ra,
        replace=True,
        n_samples=X_li.shape[0],
        random_state=random_state
    )
    X_train_bal = sparse.vstack([X_li, X_ra_up])
    y_train_bal = np.concatenate([y_li, y_ra_up])

    # Re-fit NB on the oversampled training data
    clf.fit(X_train_bal, y_train_bal)

    # 3) Return the trained vectorizer, classifier, and CV accuracy
    return vectorizer, clf, cv_accuracy


def predict_naive_bayes(vectorizer, clf, texts):
    """
    Predict labels for new texts using trained vectorizer and NB model.
    """
    X = vectorizer.transform(texts)
    return clf.predict(X)
def ensemble_predict(cvs, jd, labeled_texts, labels, alpha=0.4):
    """
    Combine IR ranking scores with NB probabilities into final label.
    Uses the original NB training logic and stores explanation per JD.

    Args:
      cvs           : list of processed CV dicts (must include 'cleaned_text').
      jd            : job-description dict from process_jd(...).
      labeled_texts : list of cleaned_texts used to train NB.
      labels        : list of 0/1 labels used to train NB.
      alpha         : weight on NB probability (0–1).
      threshold     : cutoff on combined score for RA vs LI.

    Returns:
      List of tuples (cv, ensemble_score).
    """
    # Train NB classifier and get vectorizer, clf
    vectorizer, clf, _ = train_naive_bayes(labeled_texts, labels)

    # Compute IR ranking
    ranked_ir = rank_and_filter(cvs, jd)

    # Normalize IR scores
    ir_scores = [score for _, score in ranked_ir]
    max_ir = max(ir_scores) if ir_scores else 1.0
    normalized_ir = [s / max_ir for s in ir_scores]

    # Extract texts from ranked CVs
    texts = [cv['cleaned_text'] for cv, _ in ranked_ir]

    # Get NB probabilities in same order
    probs = clf.predict_proba(vectorizer.transform(texts))

    results = []
    for (cv, _), ir_norm, (p_li, p_ra) in zip(ranked_ir, normalized_ir, probs):
        # Decide which NB prob to use based on JD title
        if jd['title'].lower() == 'research assistant':
            nb_prob = p_ra
        elif jd['title'].lower() == 'lab instructor':
            nb_prob = p_li
        else:
            nb_prob = 0  # fallback if title doesn't match expected

        # Combine scores
        ens_score = alpha * nb_prob + (1 - alpha) * ir_norm

        # Store explanations per JD title
        if 'explanation_per_jd' not in cv:
            cv['explanation_per_jd'] = {}

        cv['explanation_per_jd'][jd['title']] = {
            'ir_score':       round(ir_norm,    4),
            'nb_prob':        round(nb_prob,    4),
            'ensemble_score': round(ens_score,  4),
        }

        results.append((cv, ens_score))

    # Sort descending by ensemble score
    results.sort(key=lambda x: x[1], reverse=True)
    return results
