"""12 - Split sizes, training/validation/test performance, simple benchmarks, calibration and real examples.

* Split sizes and shares of the 51,201 matches (chronological split of notebook Section 9).
* The frozen model's scores on its own training seasons and on the validation season (recomputed with the saved
  final_model.joblib) next to the single test evaluation (read from final_test_metrics.csv; the test set is not
  re-evaluated). A large train-minus-test gap would signal overfitting.
* Benchmarks on the test seasons: always "home" (the Dummy), and a one-feature rule that predicts the most common
  training result for each tenth (decile) of xi_strength_diff. Optional: the Bet365 favourite on the Stage 1
  seasons, which needs the raw Kaggle Match.csv (odds are not part of the modelling dataset).
* Calibration on the test seasons: for each fifth of predicted P(Home), P(Draw), P(Away), the average prediction
  against how often that result happened (read from final_test_predictions.csv).
* How often the pick is right at each confidence level, and real test-season examples (report Section 24).

Run from the ML_Pipeline folder:
    python extra_experiments/12_split_performance_and_benchmarks.py [--match-csv <path to Kaggle Match.csv>]
Outputs: ml_outputs/split_summary.csv, split_performance.csv, benchmarks.csv, calibration_test.csv,
         prediction_confidence_test.csv, prediction_examples_test.csv
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _notebook_setup import load

ap = argparse.ArgumentParser()
ap.add_argument("--match-csv", default=None, help="raw Kaggle Match.csv (only for the Bet365 benchmark)")
args = ap.parse_args()
ns = load()
get_X, evaluate = ns["get_X"], ns["evaluate"]
df, tr, va, te = ns["df"], ns["train_df"], ns["valid_df"], ns["test_df"]
FEATS = pd.read_csv("ml_outputs/final_feature_list.csv").feature.tolist()
pd.set_option("display.width", 250)

# split sizes
rows = []
for name, d in (("Train (seasons 2008/09-2022/23)", tr), ("Validation (2023/24)", va), ("Test (2024/25-2025/26)", te)):
    rows.append({"Part": name, "Matches": len(d), "Share of all matches (%)": 100 * len(d) / len(df),
                 "First date": d.date.min().date(), "Last date": d.date.max().date(),
                 "Home %": 100 * (d.target_result == "H").mean(), "Draw %": 100 * (d.target_result == "D").mean(),
                 "Away %": 100 * (d.target_result == "A").mean()})
split = pd.DataFrame(rows); split.to_csv("ml_outputs/split_summary.csv", index=False)
print(split.round(1).to_string(index=False))

# frozen model on train and validation; test from the saved single evaluation
model = joblib.load("ml_outputs/final_model.joblib")
perf = []
for name, d, y in (("Train", tr, ns["y_train"]), ("Validation", va, ns["y_valid"])):
    m, _, _ = evaluate(model, get_X(d, FEATS), y)
    perf.append({"Part": name, "Accuracy": m["Accuracy"], "Macro-F1": m["Macro-F1"], "Draw F1": m["Draw F1"],
                 "Log loss": m["Log Loss"], "Brier score": m["Brier Score"]})
tm = pd.read_csv("ml_outputs/final_test_metrics.csv", index_col=0)["value"]
perf.append({"Part": "Test (saved single evaluation)", "Accuracy": tm["Accuracy"], "Macro-F1": tm["Macro-F1"],
             "Draw F1": tm["Draw F1"], "Log loss": tm["Log Loss"], "Brier score": tm["Brier Score"]})
perf = pd.DataFrame(perf); perf.to_csv("ml_outputs/split_performance.csv", index=False)
print("\n" + perf.round(4).to_string(index=False))
saved_val = pd.read_csv("ml_outputs/candidate_ranking_validation.csv").iloc[0]["Macro-F1"]
assert abs(perf.loc[1, "Macro-F1"] - saved_val) < 1e-9, "saved model does not reproduce the validation score"

# benchmarks
bm = []
yte = te.target_result.to_numpy()
always_h = np.full(len(te), "H", dtype=object)
bm.append({"Benchmark": "Always predict home win (Dummy)", "Period": "Test 2024/25-2025/26", "Matches": len(te),
           "Accuracy": accuracy_score(yte, always_h), "Macro-F1": f1_score(yte, always_h, average="macro"),
           "Draw predictions (%)": 0.0, "Log loss": np.nan})
edges = np.nanquantile(tr.xi_strength_diff, np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
majority = tr.groupby(pd.cut(tr.xi_strength_diff, edges), observed=True).target_result.agg(lambda s: s.value_counts().index[0])
for name, d in (("Train 2008/09-2022/23", tr), ("Test 2024/25-2025/26", te)):
    pred = pd.cut(d.xi_strength_diff, edges).map(majority).astype(object).fillna("H").to_numpy()
    bm.append({"Benchmark": "Most common result for each tenth of the XI strength gap", "Period": name, "Matches": len(d),
               "Accuracy": accuracy_score(d.target_result, pred), "Macro-F1": f1_score(d.target_result, pred, average="macro"),
               "Draw predictions (%)": 100 * np.mean(pred == "D"), "Log loss": np.nan})
print("\nMost common training result by tenth of xi_strength_diff (lowest to highest):", majority.tolist())
if args.match_csv:
    M = pd.read_csv(args.match_csv, usecols=["match_api_id", "B365H", "B365D", "B365A"])
    s1 = df[df.match_uid.str.startswith("S1_")].copy()
    s1["match_api_id"] = s1.match_uid.str[3:].astype("int64")
    o = s1.merge(M, on="match_api_id", how="left").dropna(subset=["B365H", "B365D", "B365A"])
    inv = 1 / o[["B365A", "B365D", "B365H"]].to_numpy(); pr = inv / inv.sum(1, keepdims=True)
    cls = np.array(["A", "D", "H"]); pred = cls[pr.argmax(1)]
    bm.append({"Benchmark": "Bet365 favourite (odds with margin removed)", "Period": "Stage 1 seasons 2008/09-2015/16",
               "Matches": len(o), "Accuracy": accuracy_score(o.target_result, pred),
               "Macro-F1": f1_score(o.target_result, pred, average="macro"), "Draw predictions (%)": 100 * np.mean(pred == "D"),
               "Log loss": log_loss(o.target_result, pr, labels=cls)})
else:
    print("Bet365 benchmark skipped (pass --match-csv to include it).")
bm.append({"Benchmark": "Final Random Forest", "Period": "Test 2024/25-2025/26", "Matches": len(te), "Accuracy": tm["Accuracy"],
           "Macro-F1": tm["Macro-F1"], "Draw predictions (%)": np.nan, "Log loss": tm["Log Loss"]})
bm = pd.DataFrame(bm); bm.to_csv("ml_outputs/benchmarks.csv", index=False)
print("\n" + bm.round(4).to_string(index=False))

# calibration on the test seasons (saved predictions)
p = pd.read_csv("ml_outputs/final_test_predictions.csv")
cal = []
for col, k in (("P(Home)", "H"), ("P(Draw)", "D"), ("P(Away)", "A")):
    b = pd.qcut(p[col], 5, labels=[f"fifth {i}" for i in range(1, 6)])
    for lab, g in p.groupby(b, observed=True):
        cal.append({"Outcome": col, "Group (by predicted probability)": lab, "Matches": len(g),
                    "Mean predicted (%)": 100 * g[col].mean(), "Actually happened (%)": 100 * (g.actual_result == k).mean()})
    cal.append({"Outcome": col, "Group (by predicted probability)": "all matches", "Matches": len(p),
                "Mean predicted (%)": 100 * p[col].mean(), "Actually happened (%)": 100 * (p.actual_result == k).mean()})
cal = pd.DataFrame(cal); cal["Gap (points)"] = cal["Actually happened (%)"] - cal["Mean predicted (%)"]
cal.to_csv("ml_outputs/calibration_test.csv", index=False)
print("\n" + cal.round(1).to_string(index=False))
print("Share of predicted classes on test (%):", (p.predicted_result.value_counts(normalize=True) * 100).round(1).to_dict())

# how often the pick is right, by how confident the model is (highest of the three probabilities)
p["max_p"] = p[["P(Home)", "P(Draw)", "P(Away)"]].max(axis=1)
band = pd.cut(p.max_p, [0, 0.4, 0.5, 0.6, 1.0], labels=["below 0.40", "0.40 to 0.50", "0.50 to 0.60", "0.60 and above"], right=False)
conf = p.groupby(band, observed=False).apply(lambda g: pd.Series({
    "Matches": len(g), "Share of matches (%)": 100 * len(g) / len(p),
    "Pick correct (%)": 100 * (g.predicted_result == g.actual_result).mean() if len(g) else np.nan}), include_groups=False)
conf.index.name = "Highest of the three probabilities"
conf.to_csv("ml_outputs/prediction_confidence_test.csv")
print("\n" + conf.round(1).to_string())
print("Highest P(Draw) for any test match: %.3f" % p["P(Draw)"].max())

# real examples for report Section 24
ex = pd.concat([
    p.loc[[p["P(Home)"].idxmax()]].assign(Example="Most confident home-win prediction"),
    p[(p.predicted_result == "H") & (p.actual_result == "A")].nlargest(1, "P(Home)").assign(Example="Confident home-win prediction that was wrong"),
    p.loc[[p["P(Away)"].idxmax()]].assign(Example="Most confident away-win prediction"),
    p.loc[[p["P(Draw)"].idxmax()]].assign(Example="Highest draw probability"),
    p.loc[[(p.max_p - 1 / 3).abs().idxmin()]].assign(Example="Closest to a three-way tie"),
])
ex = ex.merge(te[["match_uid", "league_code", "audit_home_goals", "audit_away_goals"]], on="match_uid", how="left")
ex.to_csv("ml_outputs/prediction_examples_test.csv", index=False)
print("\n" + ex[["Example", "date", "league_code", "home_team", "away_team", "P(Home)", "P(Draw)", "P(Away)", "predicted_result",
                  "actual_result", "audit_home_goals", "audit_away_goals"]].round(3).to_string(index=False))
