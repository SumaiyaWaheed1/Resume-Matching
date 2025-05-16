# tuning.py

import os
import numpy as np
from preprocessing import process_cv
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

# ——————————————————————————————————————————————————————————————
# 1) Define your labeled files and their labels (0=Lab-Instructor, 1=Research-Assistant)
# ——————————————————————————————————————————————————————————————
label_data = {
    "Umair Ahmed - CV for Lab Instructor.txt": 0,
    "Hafiz Ali Raja - CV for Lab Instructor.txt": 0,
    "Faisal Shahzad - CV for Lab Instructor.txt": 0,
    "Saad Hassan Khan - CV for Research Assistant (1).txt": 1,
    "Waqas Ahmed - CV for Lab Instructor.txt": 0,
    "Muhammad Omaid Sheikh - CV for for Lab Instructor.txt": 0,
    "Awais Anwar - CV for Lab Instructor.txt": 0,
    "Urooj Sheikh - CV for Lab Engineer.txt": 0,
    "Ghulam Jaffar - CV for Research Assistant.txt": 1,
    "Ebad Ali - CV for Research Assistant.txt": 1,
    "Sana Fatima - CV for Research Assistant.txt": 1,
    "Muhammad Tayyab Yaqoob - CV for Lab Engineer.txt": 0,
    "Haris Ahmed - CV for Research Assistant.txt": 1,
    "Faisal Nisar - CV for for Research Assistant.txt": 1,
    "Muhammad Azmi Umer - CV for Lab Instructor.txt": 0,
    "Shawana Khan - CV for Lab Instructor (1).txt": 0
}

# ——————————————————————————————————————————————————————————————
# 2) Assemble labeled_texts and labels by reading each file and preprocessing
#    Put your labeled CV files into a folder named "labeled_cvs" next to tuning.py
# ——————————————————————————————————————————————————————————————
labeled_texts = []
labels        = []

data_dir = "labeled_cvs"

for fname, lbl in label_data.items():
    path = os.path.join(data_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Expected to find {path}")
    with open(path, encoding="latin-1") as f:
        raw_text = f.read()
    cleaned = process_cv(raw_text)["cleaned_text"]
    labeled_texts.append(cleaned)
    labels.append(lbl)

# ——————————————————————————————————————————————————————————————
# 3) Build an sklearn Pipeline for TF-IDF + Naive Bayes
# ——————————————————————————————————————————————————————————————
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb',    MultinomialNB())
])

# ——————————————————————————————————————————————————————————————
# 4) Define the hyperparameter grid to search
# ——————————————————————————————————————————————————————————————
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__min_df':       [1, 2],
    'tfidf__max_df':       [0.7, 0.8],
    'tfidf__max_features': [1000, 2000],
    'nb__alpha':           [0.5, 1.0, 1.5]
}

# ——————————————————————————————————————————————————————————————
# 5) Run GridSearchCV with stratified 5-fold CV
# ——————————————————————————————————————————————————————————————
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=48)
grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring='accuracy',   # or use 'f1' or a custom scorer focusing on RA class
    cv=cv,
    n_jobs=-1,
    verbose=1
)
grid.fit(labeled_texts, labels)

print("Best parameters found:")
for k, v in grid.best_params_.items():
    print(f"  {k}: {v}")
print(f"Cross-validated accuracy: {grid.best_score_:.3f}")

# ——————————————————————————————————————————————————————————————
# 6) Optional: Evaluate the best estimator on a hold-out split
# ——————————————————————————————————————————————————————————————
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(
    labeled_texts, labels,
    test_size=0.2,
    stratify=labels,
    random_state=48
)
best_pipe = grid.best_estimator_
best_pipe.fit(X_tr, y_tr)
y_pred = best_pipe.predict(X_te)

print("\nClassification report on hold-out set:")
print(classification_report(y_te, y_pred, target_names=["Lab Instructor", "Research Assistant"]))
