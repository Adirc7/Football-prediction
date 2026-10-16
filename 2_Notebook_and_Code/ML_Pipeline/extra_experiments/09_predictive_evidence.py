"""09 - Predictive evidence: how much probability each model gave to what actually happened (validation 2023/24 only).

For every validation match the log of the probability given to the actual result is summed per model (the
log-likelihood). Differences in log-likelihood between two models are a likelihood ratio on new data, the practical
counterpart of a Bayes factor; the standard error of the paired difference shows whether a gap is larger than chance.
Same configurations as script 07.

Run from the ML_Pipeline folder:  python extra_experiments/09_predictive_evidence.py
Output: ml_outputs/predictive_evidence_validation.csv
"""
import os
import sys

import numpy as np
import pandas as pd
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
LL = {}
for label, name, params, w in CFG:
    p = dict(params)
    if name in ("LogisticRegression", "RandomForest", "DecisionTree"):
        p["class_weight"] = w; pipe = build_pipeline(name, feats, **p).fit(Xtr, ytr)
    elif name == "GradientBoosting":
        pipe = build_pipeline(name, feats, **p).fit(Xtr, ytr, clf__sample_weight=compute_sample_weight("balanced", ytr))
    else:
        pipe = build_pipeline(name, feats, **p).fit(Xtr, ytr)
    proba = pd.DataFrame(pipe.predict_proba(Xva), columns=pipe.classes_)
    pr = proba.to_numpy()[np.arange(len(yva)), [list(pipe.classes_).index(y) for y in yva]]
    LL[label] = np.log(np.clip(pr, 1e-15, 1))
N = len(yva); rf = LL["Random Forest"]
rows = []
for k, v in LL.items():
    d_rf = v - rf
    se = np.sqrt(N * d_rf.var(ddof=1)) if k != "Random Forest" else 0.0
    rows.append({"Model": k, "Total log-likelihood": v.sum(), "Mean log loss": -v.mean(),
                 "Difference vs Random Forest (log units)": d_rf.sum(), "Std. error of difference": se,
                 "z": d_rf.sum() / se if k != "Random Forest" else np.nan,
                 "Likelihood ratio vs Random Forest": np.exp(d_rf.sum())})
res = pd.DataFrame(rows).set_index("Model"); res.to_csv("ml_outputs/predictive_evidence_validation.csv")
pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda x: f"{x:,.4g}")
print("N =", N); print(res.to_string())
