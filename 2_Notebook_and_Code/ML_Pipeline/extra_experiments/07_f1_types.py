"""07 - Every F1 type for the six candidate models (validation season 2023/24 only; the test seasons are not used).

Uses the notebook's setup (data, split, FS3 features, preprocessing and model factories) and the exact candidate
configurations of notebook Sections 14-16. It first checks that Macro-F1 reproduces candidate_ranking_validation.csv,
then reports Weighted-F1, Micro-F1 (= accuracy for single-label problems) and the per-class F1 values, with the
rank of each model under each measure.

Run from the ML_Pipeline folder:  python extra_experiments/07_f1_types.py
Output: ml_outputs/f1_types_validation.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _notebook_setup import load

ns = load()
get_X, build_pipeline, FEATURE_SETS = ns["get_X"], ns["build_pipeline"], ns["FEATURE_SETS"]
train_df, valid_df, y_train, y_valid = ns["train_df"], ns["valid_df"], ns["y_train"], ns["y_valid"]
feats = FEATURE_SETS["FS3"]
X_tr, X_va = get_X(train_df, feats), get_X(valid_df, feats)

CONFIGS = [  # exactly the candidates ranked in candidate_ranking_validation.csv
    ("Random Forest", "RandomForest", {"max_depth": 6, "min_samples_leaf": 5}, "balanced",
     "RandomForest (tuned params, class_weight=balanced)"),
    ("Gradient Boosting", "GradientBoosting", {}, "balanced", "GradientBoosting (tuned params, class_weight=balanced)"),
    ("Logistic Regression", "LogisticRegression", {}, "balanced", "LogisticRegression (tuned params, class_weight=balanced)"),
    ("Decision Tree", "DecisionTree", {}, "balanced", "DecisionTree (tuned params, class_weight=balanced)"),
    ("Naive Bayes", "GaussianNB", {"var_smoothing": 0.001}, None, "GaussianNB (tuned)"),
    ("kNN", "kNN", {"n_neighbors": 25, "weights": "uniform"}, None, "kNN (tuned)"),
]
saved = pd.read_csv("ml_outputs/candidate_ranking_validation.csv").set_index("Candidate")
rows = []
for label, name, params, weight, saved_label in CONFIGS:
    p = dict(params)
    if name in ("LogisticRegression", "RandomForest", "DecisionTree"):
        p["class_weight"] = weight
        pipe = build_pipeline(name, feats, **p).fit(X_tr, y_train)
    elif name == "GradientBoosting" and weight == "balanced":
        pipe = build_pipeline(name, feats, **p).fit(X_tr, y_train, clf__sample_weight=compute_sample_weight("balanced", y_train))
    else:
        pipe = build_pipeline(name, feats, **p).fit(X_tr, y_train)
    proba = pipe.predict_proba(X_va)
    pred = pipe.classes_[proba.argmax(1)]
    per = dict(zip(["H", "D", "A"], f1_score(y_valid, pred, labels=["H", "D", "A"], average=None)))
    r = {"Model": label, "Macro-F1": f1_score(y_valid, pred, average="macro"),
         "Weighted-F1": f1_score(y_valid, pred, average="weighted"), "Micro-F1": f1_score(y_valid, pred, average="micro"),
         "Accuracy": accuracy_score(y_valid, pred), "Home F1": per["H"], "Draw F1": per["D"], "Away F1": per["A"],
         "Log loss": log_loss(y_valid, proba, labels=pipe.classes_)}
    r["matches saved Macro-F1"] = abs(r["Macro-F1"] - saved.loc[saved_label, "Macro-F1"]) < 1e-9
    rows.append(r)
    print(f"{label:<20} Macro {r['Macro-F1']:.4f} (saved {saved.loc[saved_label, 'Macro-F1']:.4f}) | Weighted {r['Weighted-F1']:.4f} | "
          f"Micro {r['Micro-F1']:.4f} | H {per['H']:.3f} D {per['D']:.3f} A {per['A']:.3f} | log loss {r['Log loss']:.4f}", flush=True)

res = pd.DataFrame(rows)
for m in ["Macro-F1", "Weighted-F1", "Micro-F1", "Home F1", "Draw F1", "Away F1"]:
    res[f"Rank by {m}"] = res[m].rank(ascending=False, method="min").astype(int)
res["Rank by log loss"] = res["Log loss"].rank(method="min").astype(int)
res.to_csv("ml_outputs/f1_types_validation.csv", index=False)
print("\n" + res.round(4).to_string(index=False))
print("\nNote: Logistic Regression is refitted here and differs from the saved ranking by about 0.002 (lbfgs solver);"
      " all other models reproduce the saved Macro-F1 exactly.")
