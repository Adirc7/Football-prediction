"""08 - Other model-selection measures for the six candidates and the Dummy baseline (validation 2023/24 only).

Balanced accuracy, Cohen's kappa, Matthews correlation (MCC), macro one-vs-rest ROC-AUC, log loss, Brier score and
the ranked probability score (RPS, which respects the order Away < Draw < Home), next to Macro-F1 and accuracy.
Same configurations as script 07.

Run from the ML_Pipeline folder:  python extra_experiments/08_other_selection_metrics.py
Output: ml_outputs/other_metrics_validation.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, log_loss,
                             matthews_corrcoef, roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _notebook_setup import load

ns = load()
get_X, build_pipeline, FS = ns["get_X"], ns["build_pipeline"], ns["FEATURE_SETS"]
tr, va, ytr, yva = ns["train_df"], ns["valid_df"], ns["y_train"], ns["y_valid"]
feats = FS["FS3"]; Xtr, Xva = get_X(tr, feats), get_X(va, feats)
CFG = [("Random Forest", "RandomForest", {"max_depth": 6, "min_samples_leaf": 5}, "balanced"),
       ("Gradient Boosting", "GradientBoosting", {}, "balanced"), ("Logistic Regression", "LogisticRegression", {}, "balanced"),
       ("Decision Tree", "DecisionTree", {}, "balanced"), ("Naive Bayes", "GaussianNB", {"var_smoothing": 0.001}, None),
       ("kNN", "kNN", {"n_neighbors": 25, "weights": "uniform"}, None), ("Dummy baseline", "Dummy", {}, None)]
ORDER = ["A", "D", "H"]           # ordered by goal difference, for the ranked probability score
rows = []
for label, name, params, w in CFG:
    p = dict(params)
    if name in ("LogisticRegression", "RandomForest", "DecisionTree"):
        p["class_weight"] = w; pipe = build_pipeline(name, feats, **p).fit(Xtr, ytr)
    elif name == "GradientBoosting":
        pipe = build_pipeline(name, feats, **p).fit(Xtr, ytr, clf__sample_weight=compute_sample_weight("balanced", ytr))
    else:
        pipe = build_pipeline(name, feats, **p).fit(Xtr, ytr)
    proba = pd.DataFrame(pipe.predict_proba(Xva), columns=pipe.classes_)[ORDER].to_numpy()
    pred = np.array(ORDER)[proba.argmax(1)]
    Y = (yva[:, None] == np.array(ORDER)[None, :]).astype(float)
    rps = np.mean(np.sum((np.cumsum(proba, 1) - np.cumsum(Y, 1))[:, :-1] ** 2, 1) / 2)
    rows.append({"Model": label, "Macro-F1": f1_score(yva, pred, average="macro"), "Accuracy": accuracy_score(yva, pred),
                 "Balanced accuracy": balanced_accuracy_score(yva, pred), "Cohen's kappa": cohen_kappa_score(yva, pred),
                 "MCC": matthews_corrcoef(yva, pred), "ROC-AUC (macro, one-vs-rest)": roc_auc_score(yva, proba, multi_class="ovr", labels=ORDER),
                 "Log loss": log_loss(yva, proba, labels=ORDER), "Brier score": np.mean(np.sum((proba - Y) ** 2, 1)), "RPS": rps})
res = pd.DataFrame(rows).set_index("Model")
res.to_csv("ml_outputs/other_metrics_validation.csv")
pd.set_option("display.width", 250); print(res.round(4).to_string())
lower = ["Log loss", "Brier score", "RPS"]
ranks = pd.DataFrame({c: res.drop("Dummy baseline")[c].rank(ascending=c in lower, method="min").astype(int) for c in res.columns})
print("\nRanks among the six models (1 = best):"); print(ranks.to_string())
