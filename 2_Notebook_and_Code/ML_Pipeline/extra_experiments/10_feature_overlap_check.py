"""10 - Does reducing feature overlap help? (validation 2023/24 only)

Nine of the 38 features are exact combinations of others (for example xi_strength_diff = home_xi_strength -
away_xi_strength), and many others are strongly correlated. Naive Bayes assumes the features are independent, so
overlap makes it count the same evidence several times and become over-confident (validation log loss 2.35).
Three remedies are tested: removing the nine derived features, decorrelating with PCA, and calibrating the
probabilities (sigmoid, time-series cross-validation inside the training seasons). The Random Forest is shown
under the same changes.

Run from the ML_Pipeline folder:  python extra_experiments/10_feature_overlap_check.py
Output: ml_outputs/overlap_fixes_validation.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _notebook_setup import load

ns = load()
get_X, mkprep, FS = ns["get_X"], ns["make_preprocessor"], ns["FEATURE_SETS"]
tr, va, ytr, yva = ns["train_df"], ns["valid_df"], ns["y_train"], ns["y_valid"]
FS3 = FS["FS3"]
DERIVED = ["home_goal_diff_avg_L5", "away_goal_diff_avg_L5", "xi_strength_diff", "home_top3_gap", "away_top3_gap",
           "home_attack_vs_away_defence", "away_attack_vs_home_defence", "midfield_strength_diff", "gk_strength_diff"]
NODUP = [f for f in FS3 if f not in DERIVED]
tscv = TimeSeriesSplit(n_splits=3)
nb = lambda: GaussianNB(var_smoothing=1e-3)
rf = lambda: RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=42)
pca = lambda: PCA(n_components=0.95, whiten=True, svd_solver="full", random_state=42)
V = [  # (label, features, pipeline factory)
    ("Naive Bayes, all 38 features (official)", FS3, lambda f: Pipeline([("prep", mkprep(f, True)), ("clf", nb())])),
    ("Naive Bayes, duplicates removed (29)", NODUP, lambda f: Pipeline([("prep", mkprep(f, True)), ("clf", nb())])),
    ("Naive Bayes, PCA (decorrelated)", FS3, lambda f: Pipeline([("prep", mkprep(f, True)), ("pca", pca()), ("clf", nb())])),
    ("Naive Bayes, calibrated (sigmoid, time-series CV)", FS3,
     lambda f: CalibratedClassifierCV(Pipeline([("prep", mkprep(f, True)), ("clf", nb())]), method="sigmoid", cv=tscv)),
    ("Naive Bayes, PCA + calibrated", FS3,
     lambda f: CalibratedClassifierCV(Pipeline([("prep", mkprep(f, True)), ("pca", pca()), ("clf", nb())]), method="sigmoid", cv=tscv)),
    ("Random Forest, all 38 features (official)", FS3, lambda f: Pipeline([("prep", mkprep(f, False)), ("clf", rf())])),
    ("Random Forest, duplicates removed (29)", NODUP, lambda f: Pipeline([("prep", mkprep(f, False)), ("clf", rf())])),
    ("Random Forest, PCA (decorrelated)", FS3, lambda f: Pipeline([("prep", mkprep(f, True)), ("pca", pca()), ("clf", rf())])),
]
rows = []
for label, feats, make in V:
    m = make(feats).fit(get_X(tr, feats), ytr)
    proba = m.predict_proba(get_X(va, feats)); pred = m.classes_[proba.argmax(1)]
    ncomp = m.named_steps["pca"].n_components_ if hasattr(m, "named_steps") and "pca" in m.named_steps else None
    rows.append({"Variant": label, "Features": len(feats), "PCA components": ncomp, "Macro-F1": f1_score(yva, pred, average="macro"),
                 "Draw F1": f1_score(yva, pred, labels=["D"], average="macro"), "Log loss": log_loss(yva, proba, labels=m.classes_),
                 "Max probability (mean)": proba.max(1).mean()})
res = pd.DataFrame(rows); res.to_csv("ml_outputs/overlap_fixes_validation.csv", index=False)
pd.set_option("display.width", 250); print(res.round(4).to_string(index=False))
