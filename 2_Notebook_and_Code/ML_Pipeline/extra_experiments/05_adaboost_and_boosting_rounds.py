"""05 - AdaBoost check and boosting-rounds curve (validation only; the test seasons are not used).

Same data, features, split, preprocessing, class weighting and decision rule as notebook Section 26:
FS3 (38 features), train season_start <= 2022, validation 2023, balanced sample weights,
tuning with GridSearchCV over a 3-fold TimeSeriesSplit inside the training period, and the frozen
Random Forest configuration retrained in the same run as the reference.

Part A: AdaBoost (SAMME) with shallow trees, tuned.
Part B: how training and validation Macro-F1 change as boosting adds rounds (AdaBoost and Gradient
Boosting), next to the Random Forest's training/validation gap. Shows whether extra rounds keep
improving on new matches or only on the training matches.

Run from the ML_Pipeline folder:  python extra_experiments/05_adaboost_and_boosting_rounds.py
Outputs: ml_outputs/adaboost_validation.csv, ml_outputs/boosting_rounds_curve.csv
"""
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
print = __import__("functools").partial(print, flush=True)
print("Python", sys.version.split()[0], "| scikit-learn", sklearn.__version__)

OUT_DIR = "ml_outputs"
DATA_PATH = "European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv"
FEATS = pd.read_csv(os.path.join(OUT_DIR, "final_feature_list.csv")).feature.tolist()
CAT = ["league_code"]
NUM = [f for f in FEATS if f not in CAT]
df = pd.read_csv(DATA_PATH, low_memory=False, parse_dates=["date"]).sort_values("date", kind="stable")
tr, va = df[df.season_start <= 2022], df[df.season_start == 2023]
X_tr, y_tr, X_va, y_va = tr[FEATS], tr.target_result.to_numpy(), va[FEATS], va.target_result.to_numpy()
CLASSES = np.array(["A", "D", "H"])
W_TR = compute_sample_weight("balanced", y_tr)
tscv = TimeSeriesSplit(n_splits=3)
print(f"Training {len(tr):,} matches | validation {len(va):,} matches | {len(FEATS)} features")


def prep():
    num = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])


def scores(y, proba):
    pred = CLASSES[proba.argmax(1)]
    return {"Macro-F1": f1_score(y, pred, average="macro"), "Accuracy": accuracy_score(y, pred),
            "Draw F1": f1_score(y, pred, labels=["D"], average="macro"), "Log Loss": log_loss(y, proba, labels=CLASSES)}


# ---------------------------------------------------------------- Part A
rows = []
frozen = joblib.load(os.path.join(OUT_DIR, "final_model.joblib"))
t0 = time.time()
rf = clone(frozen).fit(X_tr, y_tr)
rf_fit = time.time() - t0
rf_va, rf_tr = scores(y_va, rf.predict_proba(X_va)), scores(y_tr, rf.predict_proba(X_tr))
print(f"Reference Random Forest fitted in {rf_fit:.0f}s")
rows.append({"Model": "Random Forest (frozen configuration, reference)", **rf_va, "Fit time (s)": round(rf_fit, 1), "Best params": "-"})
REF = rf_va["Macro-F1"]

ada = Pipeline([("prep", prep()), ("clf", AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                                                             n_estimators=200, random_state=42))])
grid = {"clf__estimator__max_depth": [1, 3], "clf__learning_rate": [0.1, 0.5]}
t0 = time.time()
gs = GridSearchCV(ada, grid, scoring="f1_macro", cv=tscv, n_jobs=1).fit(X_tr, y_tr, clf__sample_weight=W_TR)
ada_fit = time.time() - t0
best = gs.best_estimator_
params = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
ada_va = scores(y_va, best.predict_proba(X_va))
rows.append({"Model": "AdaBoost (balanced, tuned)", **ada_va, "Fit time (s)": round(ada_fit, 1), "Best params": str(params)})
res = pd.DataFrame(rows)
res["Gap to frozen RF"] = res["Macro-F1"] - REF
res.to_csv(os.path.join(OUT_DIR, "adaboost_validation.csv"), index=False)
print(res.round(4).to_string(index=False))
gain = ada_va["Macro-F1"] - REF
print(f"\nAdaBoost gain over the reference Random Forest: {gain:+.4f}. Decision rule (gain > 0.01): "
      f"{'met' if gain > 0.01 else 'not met - the Random Forest stays the final model'}.")

# ---------------------------------------------------------------- Part B
STAGES = [10, 25, 50, 100, 200, 300]
curve = []
print('Part B: fitting the curves')
ada_c = clone(best).set_params(clf__n_estimators=max(STAGES))
ada_c.fit(X_tr, y_tr, clf__sample_weight=W_TR)
gb_c = Pipeline([("prep", prep()), ("clf", GradientBoostingClassifier(n_estimators=max(STAGES), learning_rate=0.1, max_depth=3,
                                                                      subsample=0.8, random_state=42))])
gb_c.fit(X_tr, y_tr, clf__sample_weight=W_TR)
for name, pipe in (("AdaBoost (best settings)", ada_c), ("Gradient Boosting (depth 3, learning rate 0.1)", gb_c)):
    Xt_tr, Xt_va = pipe.named_steps["prep"].transform(X_tr), pipe.named_steps["prep"].transform(X_va)
    clf = pipe.named_steps["clf"]
    for k, (p_tr, p_va) in enumerate(zip(clf.staged_predict_proba(Xt_tr), clf.staged_predict_proba(Xt_va)), start=1):
        if k in STAGES:
            s_tr, s_va = scores(y_tr, p_tr), scores(y_va, p_va)
            curve.append({"Model": name, "Rounds": k, "Train Macro-F1": s_tr["Macro-F1"], "Validation Macro-F1": s_va["Macro-F1"],
                          "Gap (train - validation)": s_tr["Macro-F1"] - s_va["Macro-F1"], "Validation log loss": s_va["Log Loss"]})
curve.append({"Model": "Random Forest (frozen configuration)", "Rounds": 200, "Train Macro-F1": rf_tr["Macro-F1"],
              "Validation Macro-F1": rf_va["Macro-F1"], "Gap (train - validation)": rf_tr["Macro-F1"] - rf_va["Macro-F1"],
              "Validation log loss": rf_va["Log Loss"]})
cur = pd.DataFrame(curve)
cur.to_csv(os.path.join(OUT_DIR, "boosting_rounds_curve.csv"), index=False)
print("\n" + cur.round(4).to_string(index=False))
