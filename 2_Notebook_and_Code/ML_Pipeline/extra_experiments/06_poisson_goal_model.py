"""06 - Poisson goal model: the classic statistical approach, compared with the Random Forest (validation only).

The football literature modelled match results with goal-count models long before machine learning
(Maher 1982; Dixon and Coles 1997; Karlis and Ntzoufras 2009, report Section 5.4). Each team's goals are
treated as a Poisson count with an expected value (lambda); the two counts give the chance of every
scoreline, and adding the scorelines gives P(home win), P(draw) and P(away win).

Part A  the idea on an "average" training match (only the mean home and away goals are used)
Part B  Poisson regression: lambda_home and lambda_away are predicted from the same 38 pre-match
        features as the frozen Random Forest (FS3), fitted on the training seasons only.
        Goals (audit_home_goals, audit_away_goals) are used ONLY as training targets, exactly like
        target_result; they never enter X.
Part C  scoreline grid -> H/D/A probabilities, two decision rules (most probable result, and the
        prior-corrected rule of notebook Section 26.5, the decision-time version of class weighting),
        compared with the frozen Random Forest configuration retrained in the same run.

Same data, split, feature list and preprocessing factory as the notebook (setup cells are executed from
Machine_Learning_Pipeline.ipynb). The test seasons are not used. Decision rule fixed in advance (as in
Sections 25-26): the Poisson model would only be considered if its validation Macro-F1 beat the Random
Forest by more than 0.01.

Run from the ML_Pipeline folder:  python extra_experiments/06_poisson_goal_model.py
Outputs: ml_outputs/poisson_validation.csv, ml_outputs/poisson_goal_checks.csv,
         ml_outputs/poisson_coefficients.csv, ml_outputs/figures/poisson_scoreline_grid.png
"""
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_poisson_deviance)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
print = __import__("functools").partial(print, flush=True)
print("Python", sys.version.split()[0], "| scikit-learn", sklearn.__version__)

# ---------------------------------------------------------------- notebook setup (data, split, features)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _notebook_setup import load          # runs the notebook's own setup cells
ns = load()
get_X, build_pipeline, make_preprocessor = ns["get_X"], ns["build_pipeline"], ns["make_preprocessor"]
train_df, valid_df, y_tr, y_va = ns["train_df"], ns["valid_df"], ns["y_train"], ns["y_valid"]
FEATS = ns["FEATURE_SETS"]["FS3"]
assert FEATS == pd.read_csv("ml_outputs/final_feature_list.csv").feature.tolist(), "feature list differs from the frozen model"
assert train_df["date"].is_monotonic_increasing, "training rows must be in time order for TimeSeriesSplit"
X_tr, X_va = get_X(train_df, FEATS), get_X(valid_df, FEATS)
g_tr_h, g_tr_a = train_df["audit_home_goals"].to_numpy(float), train_df["audit_away_goals"].to_numpy(float)
g_va_h, g_va_a = valid_df["audit_home_goals"].to_numpy(float), valid_df["audit_away_goals"].to_numpy(float)
OUT, FIG = "ml_outputs", os.path.join("ml_outputs", "figures")
os.makedirs(FIG, exist_ok=True)
CLASSES = np.array(["A", "D", "H"])                 # alphabetical, as scikit-learn orders them
PRIOR = pd.Series(y_tr).value_counts(normalize=True).reindex(CLASSES).to_numpy()
MAXG = 10                                           # scorelines 0..10 per team (P(more) is negligible)
print(f"Training {len(train_df):,} | validation {len(valid_df):,} | {len(FEATS)} features")


def outcome_probs(lam_h, lam_a):
    """P(A), P(D), P(H) for each match from two independent Poisson goal counts."""
    k = np.arange(MAXG + 1)
    ph = poisson.pmf(k[None, :], np.asarray(lam_h)[:, None])           # n x 11
    pa = poisson.pmf(k[None, :], np.asarray(lam_a)[:, None])
    grid = ph[:, :, None] * pa[:, None, :]                               # n x home x away
    p_h = np.tril(np.ones((MAXG + 1, MAXG + 1)), -1)                     # home goals > away goals
    out = np.stack([(grid * p_h.T).sum((1, 2)), np.einsum("nii->n", grid), (grid * p_h).sum((1, 2))], 1)
    return out / out.sum(1, keepdims=True), grid


def scores(y, proba, pred):
    Y = (y[:, None] == CLASSES[None, :]).astype(float)
    order = [0, 1, 2]                                                    # A < D < H by goal difference
    rps = np.mean(np.sum((np.cumsum(proba[:, order], 1) - np.cumsum(Y[:, order], 1))[:, :-1] ** 2, 1) / 2)
    per = dict(zip(["H", "D", "A"], f1_score(y, pred, labels=["H", "D", "A"], average=None)))
    return {"Macro-F1": f1_score(y, pred, average="macro"), "Accuracy": accuracy_score(y, pred),
            "Home F1": per["H"], "Draw F1": per["D"], "Away F1": per["A"],
            "Log loss": log_loss(y, proba, labels=CLASSES), "Brier score": np.mean(np.sum((proba - Y) ** 2, 1)),
            "RPS": rps, "Draws predicted (%)": 100 * np.mean(pred == "D")}


# ---------------------------------------------------------------- Part A: the idea on an average match
mh, ma = g_tr_h.mean(), g_tr_a.mean()
pA, grid_avg = outcome_probs([mh], [ma])
actual = pd.Series(y_tr).value_counts(normalize=True).reindex(CLASSES).to_numpy()
checks = [
    {"Check": "Training seasons: mean home goals", "Value": mh},
    {"Check": "Training seasons: mean away goals", "Value": ma},
    {"Check": "Average match, Poisson P(home win)", "Value": pA[0, 2]},
    {"Check": "Average match, Poisson P(draw)", "Value": pA[0, 1]},
    {"Check": "Average match, Poisson P(away win)", "Value": pA[0, 0]},
    {"Check": "Training seasons: actual home-win share", "Value": actual[2]},
    {"Check": "Training seasons: actual draw share", "Value": actual[1]},
    {"Check": "Training seasons: actual away-win share", "Value": actual[0]},
    {"Check": "Training seasons: variance/mean of home goals (1 = exactly Poisson)", "Value": g_tr_h.var() / mh},
    {"Check": "Training seasons: variance/mean of away goals (1 = exactly Poisson)", "Value": g_tr_a.var() / ma},
]
print("\nPart A - average training match: lambda_home %.3f, lambda_away %.3f -> P(H) %.3f, P(D) %.3f, P(A) %.3f "
      "(actual %.3f / %.3f / %.3f)" % (mh, ma, pA[0, 2], pA[0, 1], pA[0, 0], actual[2], actual[1], actual[0]))

fig, ax = plt.subplots(figsize=(6.2, 5.0))
G = grid_avg[0, :6, :6] * 100
ax.imshow(G, cmap="Blues")
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{G[i, j]:.1f}", ha="center", va="center", fontsize=9, color="white" if G[i, j] > 7 else "black",
                fontweight="bold" if i == j else "normal")
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xlabel("Away team goals"); ax.set_ylabel("Home team goals")
ax.set_title(f"Chance of each score (%), average match\nhome {mh:.2f} goals, away {ma:.2f} goals: "
             f"H {pA[0, 2]:.1%}, D {pA[0, 1]:.1%}, A {pA[0, 0]:.1%}", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "poisson_scoreline_grid.png"), dpi=150); plt.close("all")

# ---------------------------------------------------------------- Part B: Poisson regression on the 38 features
tscv = TimeSeriesSplit(n_splits=3)
GRID = {"clf__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
models, alphas, fit_s = {}, {}, 0.0
for side, y_goals in (("home", g_tr_h), ("away", g_tr_a)):
    pipe = Pipeline([("prep", make_preprocessor(FEATS, True)), ("clf", PoissonRegressor(max_iter=1000))])
    t0 = time.time()
    gs = GridSearchCV(pipe, GRID, scoring="neg_mean_poisson_deviance", cv=tscv, n_jobs=1).fit(X_tr, y_goals)
    fit_s += time.time() - t0
    models[side], alphas[side] = gs.best_estimator_, gs.best_params_["clf__alpha"]
    print(f"{side} goals: best alpha {alphas[side]} (time-series CV mean Poisson deviance {-gs.best_score_:.4f})")
lam_h, lam_a = models["home"].predict(X_va), models["away"].predict(X_va)
proba_pois, grid_va = outcome_probs(lam_h, lam_a)

# ---------------------------------------------------------------- Part C: compare with the Random Forest
rows = []
t0 = time.time()
rf = build_pipeline("RandomForest", FEATS, max_depth=6, min_samples_leaf=5, class_weight="balanced").fit(X_tr, y_tr)
rf_s = time.time() - t0
proba_rf = pd.DataFrame(rf.predict_proba(X_va), columns=rf.classes_)[list(CLASSES)].to_numpy()
pred_rf = CLASSES[proba_rf.argmax(1)]
rows.append({"Model": "Random Forest (frozen configuration, reference)", **scores(y_va, proba_rf, pred_rf),
             "Fit time (s)": round(rf_s, 1), "Settings": "200 trees, depth 6, leaf 5, balanced"})
pred_max = CLASSES[proba_pois.argmax(1)]
pred_prior = CLASSES[(proba_pois / PRIOR).argmax(1)]
setting = f"alpha home {alphas['home']}, away {alphas['away']}"
rows.append({"Model": "Poisson goal model (most probable result)", **scores(y_va, proba_pois, pred_max),
             "Fit time (s)": round(fit_s, 1), "Settings": setting})
rows.append({"Model": "Poisson goal model (prior-corrected decision)", **scores(y_va, proba_pois, pred_prior),
             "Fit time (s)": round(fit_s, 1), "Settings": setting + "; same probabilities, decision divided by class share"})
dummy = np.tile(PRIOR, (len(y_va), 1))
rows.append({"Model": "Dummy baseline (training class shares)", **scores(y_va, dummy, np.full(len(y_va), "H", dtype=object)),
             "Fit time (s)": 0.0, "Settings": "-"})
res = pd.DataFrame(rows)
REF = res.loc[0, "Macro-F1"]
res["Gap to frozen RF (Macro-F1)"] = res["Macro-F1"] - REF

# paired bootstrap of the Macro-F1 difference (Poisson prior-corrected minus Random Forest), validation season
rng = np.random.default_rng(42)
idx = np.arange(len(y_va)); diffs = []
for _ in range(1000):
    b = rng.choice(idx, len(idx), replace=True)
    diffs.append(f1_score(y_va[b], pred_prior[b], average="macro") - f1_score(y_va[b], pred_rf[b], average="macro"))
lo, hi = np.percentile(diffs, [2.5, 97.5])
res["95% CI of gap (paired bootstrap)"] = ["-", "-", f"{lo:+.3f} to {hi:+.3f}", "-"]
res.to_csv(os.path.join(OUT, "poisson_validation.csv"), index=False)
pd.set_option("display.width", 250)
print("\n" + res.drop(columns=["Settings"]).round(4).to_string(index=False))
gain = res.loc[2, "Macro-F1"] - REF
print(f"\nPrior-corrected Poisson minus Random Forest: {gain:+.4f} (95% CI {lo:+.3f} to {hi:+.3f}). "
      f"Decision rule (gain > 0.01): {'met' if gain > 0.01 else 'not met - the Random Forest stays the final model'}.")

# goal-level checks on the validation season (things only a goal model can be checked on)
best_score = grid_va.reshape(len(y_va), -1).argmax(1)
bh, ba = np.divmod(best_score, MAXG + 1)
null_dev_h = mean_poisson_deviance(g_va_h, np.full(len(g_va_h), mh))
null_dev_a = mean_poisson_deviance(g_va_a, np.full(len(g_va_a), ma))
checks += [
    {"Check": "Validation: mean predicted home goals", "Value": lam_h.mean()},
    {"Check": "Validation: actual mean home goals", "Value": g_va_h.mean()},
    {"Check": "Validation: mean predicted away goals", "Value": lam_a.mean()},
    {"Check": "Validation: actual mean away goals", "Value": g_va_a.mean()},
    {"Check": "Validation: home-goal deviance explained vs training mean (D2)", "Value": 1 - mean_poisson_deviance(g_va_h, lam_h) / null_dev_h},
    {"Check": "Validation: away-goal deviance explained vs training mean (D2)", "Value": 1 - mean_poisson_deviance(g_va_a, lam_a) / null_dev_a},
    {"Check": "Validation: mean predicted P(draw)", "Value": proba_pois[:, 1].mean()},
    {"Check": "Validation: actual draw share", "Value": np.mean(y_va == "D")},
    {"Check": "Validation: mean predicted P(0-0)", "Value": grid_va[:, 0, 0].mean()},
    {"Check": "Validation: actual 0-0 share", "Value": np.mean((g_va_h == 0) & (g_va_a == 0))},
    {"Check": "Validation: highest single P(draw) for any match", "Value": proba_pois[:, 1].max()},
    {"Check": "Validation: matches where draw is the most probable result (%)", "Value": 100 * np.mean(pred_max == "D")},
    {"Check": "Validation: most likely exact score correct (%)", "Value": 100 * np.mean((bh == g_va_h) & (ba == g_va_a))},
]
chk = pd.DataFrame(checks)
chk.to_csv(os.path.join(OUT, "poisson_goal_checks.csv"), index=False)
print("\n" + chk.round(4).to_string(index=False))

# coefficients: numeric features are standardised, so exp(coef) is the multiplier on expected goals for a
# one-standard-deviation increase; league rows are one-hot indicators (shrunk towards zero by the penalty alpha)
names = models["home"].named_steps["prep"].get_feature_names_out()
coef = pd.DataFrame({"feature": names, "home_goals_coef": models["home"].named_steps["clf"].coef_,
                     "away_goals_coef": models["away"].named_steps["clf"].coef_})
coef["home_goals_multiplier"] = np.exp(coef.home_goals_coef)
coef["away_goals_multiplier"] = np.exp(coef.away_goals_coef)
coef.to_csv(os.path.join(OUT, "poisson_coefficients.csv"), index=False)
print("\nLargest coefficients in the home-goals model (saved in full to poisson_coefficients.csv):")
print(coef.reindex(coef.home_goals_coef.abs().sort_values(ascending=False).index).head(8).round(4).to_string(index=False))
