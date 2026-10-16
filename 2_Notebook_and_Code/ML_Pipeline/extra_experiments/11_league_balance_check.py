"""11 - League balance across the training, validation and test seasons.

Answers three questions:
1. Does every league appear in every part of the split, and in similar proportions? (counts and shares)
2. Would giving every league the same total weight in training change the result? The frozen Random Forest
   configuration is retrained twice on the training seasons, as frozen and with equal league weights, and compared
   on the validation season only (the test seasons are not used for this).
3. Is a league's test score related to how many of its matches were in training? (per-league results read from the
   saved test predictions; nothing is re-predicted)

Run from the ML_Pipeline folder:  python extra_experiments/11_league_balance_check.py
Outputs: ml_outputs/league_split_counts.csv, ml_outputs/league_weight_validation.csv, ml_outputs/league_test_results.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _notebook_setup import load

ns = load()
get_X, build_pipeline, FS = ns["get_X"], ns["build_pipeline"], ns["FEATURE_SETS"]
tr, va, te, ytr, yva = ns["train_df"], ns["valid_df"], ns["test_df"], ns["y_train"], ns["y_valid"]
feats = FS["FS3"]
pd.set_option("display.width", 250)

# 1. counts and shares
cnt = pd.DataFrame({"Train": tr.league_code.value_counts(), "Validation": va.league_code.value_counts(),
                    "Test": te.league_code.value_counts()}).fillna(0).astype(int).sort_index()
for c in ["Train", "Validation", "Test"]:
    cnt[f"{c} share (%)"] = 100 * cnt[c] / cnt[c].sum()
cnt.index.name = "league_code"
cnt.to_csv("ml_outputs/league_split_counts.csv")
print(cnt.round(1).to_string())
assert (cnt[["Train", "Validation", "Test"]] > 0).all().all(), "a league is missing from one part of the split"

# 2. equal league weights (validation only)
lw = tr.league_code.map(len(tr) / (tr.league_code.nunique() * tr.league_code.value_counts())).to_numpy()
rows, per = [], []
for name, kw in (("As frozen (class weights only)", {}), ("Plus equal league weights", {"clf__sample_weight": lw})):
    m = build_pipeline("RandomForest", feats, max_depth=6, min_samples_leaf=5, class_weight="balanced").fit(get_X(tr, feats), ytr, **kw)
    pr = m.predict_proba(get_X(va, feats)); pred = m.classes_[pr.argmax(1)]
    rows.append({"Training": name, "League": "All", "Macro-F1": f1_score(yva, pred, average="macro"),
                 "Accuracy": accuracy_score(yva, pred), "Log loss": log_loss(yva, pr, labels=m.classes_)})
    for lg in sorted(va.league_code.unique()):
        k = (va.league_code == lg).to_numpy()
        rows.append({"Training": name, "League": lg, "Macro-F1": f1_score(yva[k], pred[k], average="macro"),
                     "Accuracy": accuracy_score(yva[k], pred[k]), "Log loss": np.nan})
res = pd.DataFrame(rows)
res.to_csv("ml_outputs/league_weight_validation.csv", index=False)
pv = res.pivot(index="League", columns="Training", values="Macro-F1")
pv["Change"] = pv["Plus equal league weights"] - pv["As frozen (class weights only)"]
print("\nValidation Macro-F1 (2023/24):"); print(pv.round(4).to_string())

# 3. per-league test results from the saved predictions
pred = pd.read_csv("ml_outputs/final_test_predictions.csv").merge(te[["match_uid", "league_code"]], on="match_uid", how="left")
assert pred.league_code.notna().all() and len(pred) == len(te)
lt = []
for lg, g in pred.groupby("league_code"):
    lt.append({"league_code": lg, "Training matches": int(cnt.loc[lg, "Train"]), "Test matches": len(g),
               "Accuracy": accuracy_score(g.actual_result, g.predicted_result),
               "Macro-F1": f1_score(g.actual_result, g.predicted_result, average="macro"),
               "Home-win share": (g.actual_result == "H").mean(), "Draw share": (g.actual_result == "D").mean()})
lt = pd.DataFrame(lt)
lt.to_csv("ml_outputs/league_test_results.csv", index=False)
print("\nTest seasons (saved predictions):"); print(lt.round(3).to_string(index=False))
print("Correlation between training matches and test Macro-F1 across leagues: %.2f" % lt["Training matches"].corr(lt["Macro-F1"]))
