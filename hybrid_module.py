import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy import sparse
from ranking_module import rank_and_filter


def train_naive_bayes(labeled_texts, labels, test_size=0.2, random_state=48):
    """
    Train a TF-IDF + Naive Bayes classifier on labeled texts with oversampling
    and bigram feature selection to mitigate overfitting.
    """
    # 1) Vectorize with unigrams + bigrams, prune rare/common terms, limit vocab size
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        max_features=2000
    )
    X = vectorizer.fit_transform(labeled_texts)

    # 2) Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # 3) Oversample the minority class (Research Assistant=1) in the TRAIN set only
    y_train = np.array(y_train)
    X_train = sparse.csr_matrix(X_train)
    # Masks
    ra_mask = (y_train == 1)
    li_mask = (y_train == 0)
    X_ra, y_ra = X_train[ra_mask], y_train[ra_mask]
    X_li, y_li = X_train[li_mask], y_train[li_mask]
    # Upsample
    X_ra_up, y_ra_up = resample(
        X_ra, y_ra,
        replace=True,
        n_samples=X_li.shape[0],
        random_state=random_state
    )
    # Recombine balanced training set
    X_train_bal = sparse.vstack([X_li, X_ra_up])
    y_train_bal = np.concatenate([y_li, y_ra_up])

    # 4) Train Naive Bayes with Laplace smoothing
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train_bal, y_train_bal)

    # 5) Evaluate on hold-out test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return vectorizer, clf, accuracy


def predict_naive_bayes(vectorizer, clf, texts):
    """
    Predict labels for new texts using trained vectorizer and NB model.
    """
    X = vectorizer.transform(texts)
    return clf.predict(X)


def ensemble_predict(cvs, jd, labeled_texts, labels, alpha=0.4, threshold=0.45):
    """
    Combine IR ranking scores with NB probabilities into final label.
    Uses oversampled NB and bigram TF-IDF to reduce bias and overfitting.

    Args:
      cvs           : list of processed CV dicts (must include 'cleaned_text').
      jd            : job-description dict from process_jd(...).
      labeled_texts : list of cleaned_texts used to train NB.
      labels        : list of 0/1 labels used to train NB.
      alpha         : weight on NB probability (0–1).
      threshold     : cutoff on combined score for RA vs LI.

    Returns:
      List of tuples (cv, ensemble_label).
    """
    # Retrain NB with oversampling and bigrams
    vectorizer, clf, _ = train_naive_bayes(labeled_texts, labels)
    # Compute IR ranking
    ranked_ir = rank_and_filter(cvs, jd)
    # Normalize IR scores
    ir_scores = [score for _, score in ranked_ir]
    max_ir = max(ir_scores) if ir_scores else 1.0
    normalized_ir = [s / max_ir for s in ir_scores]

    # NB probabilities on the same order
    texts = [cv['cleaned_text'] for cv, _ in ranked_ir]
    probs = clf.predict_proba(vectorizer.transform(texts))

    results = []
    for (cv, _), ir_norm, (p_li, p_ra) in zip(ranked_ir, normalized_ir, probs):
        ens_score = alpha * p_ra + (1 - alpha) * ir_norm
        label = 'Research Assistant' if ens_score > threshold else 'Lab Instructor'
        cv['explanation'].update({
            'ir_score': round(ir_norm, 4),
            'nb_prob_ra': round(p_ra, 4),
            'ensemble_score': round(ens_score, 4),
        })
        results.append((cv, label))
    return results
