# Extra experiments (cited in the report)

These scripts produced results that the report quotes but that are not cells of `Machine_Learning_Pipeline.ipynb`
(script 06 is also run, unchanged, as notebook Section 27). They only read the data; none of them changes the official
dataset, the frozen model or the saved test predictions, and none of them evaluates a new model on the test seasons.

| Script | What it answers | Output | Report |
|---|---|---|---|
| `01_build_new_prematch_features.py` | Builds rest days, card discipline and player form for Stage 2 matches, using **earlier matches only** | `new_features_tm.csv` | Section 18.3 |
| `02_test_new_features.py` | Do those features improve the frozen model? Includes a 200-starter brute-force leakage check | `../ml_outputs/new_feature_experiment.csv` | Section 18.3 (row 4), Section 25 |
| `03_fifa_vs_market_value_overlap.py` | Do FIFA ratings and market values rank teams the same way? (2012-2015 matches present in both sources) | printed (xi_strength_diff r = 0.937) | Section 25 |
| `04_same_season_ranking.py` | Alternative Stage 2 strengths ranked within the same season (transfer-fee inflation check) | `../ml_outputs/stage2_strength_same_season_ranking.csv` | Section 18.3 (row 3, via notebook Section 25) |
| `05_adaboost_and_boosting_rounds.py` | Does AdaBoost (the original boosting method) beat the Random Forest? And how do training and validation Macro-F1 change as boosting adds rounds? Validation only, same setup as notebook Section 26 | `../ml_outputs/adaboost_validation.csv`, `../ml_outputs/boosting_rounds_curve.csv` | Section 18.5 |
| `06_poisson_goal_model.py` | The classic statistical approach: does a Poisson goal model (expected goals per team, scoreline grid, then H/D/A) beat the Random Forest? Validation only; also run as notebook Section 27 | `../ml_outputs/poisson_validation.csv`, `poisson_goal_checks.csv`, `poisson_coefficients.csv`, `figures/poisson_scoreline_grid.png` | Section 18.8 |
| `07_f1_types.py` | Macro-, Weighted- and Micro-F1 and the per-class F1 for the six candidates, with each model's rank under each | `../ml_outputs/f1_types_validation.csv` | Section 16.1 |
| `08_other_selection_metrics.py` | Balanced accuracy, Cohen's kappa, MCC, ROC-AUC, log loss, Brier score and RPS for the six candidates | `../ml_outputs/other_metrics_validation.csv` | Section 18.6 |
| `09_predictive_evidence.py` | Log-likelihood each model gave to the actual validation results (likelihood ratio vs the Random Forest) | `../ml_outputs/predictive_evidence_validation.csv` | Section 18.6 |
| `10_feature_overlap_check.py` | Does removing duplicate features, PCA or calibration fix Naive Bayes' over-confidence? Same changes for the Random Forest | `../ml_outputs/overlap_fixes_validation.csv` | Section 18.7 |
| `11_league_balance_check.py` | Is every league in every part of the split? Does equal league weighting change the result (validation)? Per-league test results from the saved predictions | `../ml_outputs/league_split_counts.csv`, `league_weight_validation.csv`, `league_test_results.csv` | Section 12 |
| `12_split_performance_and_benchmarks.py` | Split shares; the frozen model's training/validation/test scores; simple benchmarks (always home, XI-gap rule, Bet365 favourite); calibration, accuracy by confidence level and real examples from the saved test predictions | `../ml_outputs/split_summary.csv`, `split_performance.csv`, `benchmarks.csv`, `calibration_test.csv`, `prediction_confidence_test.csv`, `prediction_examples_test.csv` | Sections 19 and 24 |

## Folder layout
```
<project>/
  Build_Pipeline/Build_Master_Dataset.py        (scripts 03 and 04 import it)
  Official_Dataset_Files/Stage1_..._Causal_v2.csv (script 03)
  ML_Pipeline/
    European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv
    ml_outputs/final_model.joblib, final_feature_list.csv
    extra_experiments/   <- these scripts
```

## Run (from this folder)
```
pip install -r ../requirements.txt
python 01_build_new_prematch_features.py --stage2-dir "<path>/csv/stage_2"
python 02_test_new_features.py          --stage2-dir "<path>/csv/stage_2"
python 03_fifa_vs_market_value_overlap.py --stage2-dir "<path>/csv/stage_2"
python 04_same_season_ranking.py         --stage2-dir "<path>/csv/stage_2"
```
Scripts 05 to 12 run from the `ML_Pipeline` folder and need no raw files, for example `python extra_experiments/05_adaboost_and_boosting_rounds.py` (about 10 minutes).
Scripts 06 to 12 use `_notebook_setup.py`, which runs the notebook's own setup cells (data, split, feature sets, preprocessing and model
factories), so they use exactly the same data and pipelines as the notebook. Script 12 adds the Bet365 benchmark only when given the raw
Kaggle file: `python extra_experiments/12_split_performance_and_benchmarks.py --match-csv <path>/Match.csv`.
`csv/stage_2` is the raw Transfermarkt folder (games.csv, appearances.csv, game_lineups.csv, player_valuations.csv, competitions.csv).
Script 02 needs the output of script 01. Script 02 can take several minutes; script 04 rebuilds Stage 2 and takes longer.

## Results when re-run (scikit-learn 1.9)
* 01 and 04 reproduce the saved files exactly.
* 02: leakage check 200/200 identical. No feature group improves the model beyond noise: best validation gain +0.004
  (all new features), best test gain +0.004 (player form), and every test 95% interval includes 0.
* 03: 9,046 matches matched one-to-one; xi_strength_diff correlates r = 0.937 (Spearman 0.932) between the two sources.

Scripts 02 and notebook Section 25 retrain the frozen configuration with scikit-learn 1.9, so their baselines differ
slightly from the notebook's main run (scikit-learn 1.8): 0.472 instead of 0.473 (full history) and 0.470 instead of 0.471
(modern era only). Comparisons inside each script always use the same version.

## Script 05 results (Python 3.13, scikit-learn 1.9.1, same versions as notebook Section 26)
* AdaBoost (balanced sample weights, tuned: tree depth 1, learning rate 0.5, 200 rounds): validation Macro-F1 0.482, 0.013 below the Random Forest retrained in the same run (0.495); Draw F1 0.320; log loss 1.059. The 0.01 rule is not met.
* Gradient Boosting (depth 3, learning rate 0.1): from 100 to 300 rounds, training Macro-F1 rises from 0.496 to 0.540 while validation Macro-F1 stays between 0.495 and 0.499.
* AdaBoost: validation log loss rises with every extra round, from 1.032 after 10 rounds to 1.060 after 300.

## Scripts 06 to 12 results (Python 3.13, scikit-learn 1.8.0, the version of the notebook's main run)
* **06 Poisson goal model** (validation 2023/24): Macro-F1 0.405 when the most probable result is chosen (it never predicts a draw: the
  highest draw probability for any match is 0.292) and 0.453 with the prior-corrected decision of notebook Section 26.5, against 0.492 for
  the Random Forest retrained in the same run (gap −0.039, paired bootstrap 95% CI −0.054 to −0.024). The 0.01 rule is not met.
  Its probabilities are slightly better than the Random Forest's: log loss 0.969 vs 0.983, Brier 0.576 vs 0.587, RPS 0.192 vs 0.196.
  On an average training match (home 1.55, away 1.20 goals) the scoreline grid gives H 45.4%, D 25.1%, A 29.5% (actual 45.3 / 24.8 / 29.9).
* **07** reproduces the saved validation Macro-F1 exactly for five of the six candidates; Logistic Regression differs by about 0.002
  (0.486 vs 0.485). The Random Forest ranks first on Macro-F1, third on Weighted-F1 and fifth on Micro-F1 (= accuracy).
* **08** The Random Forest is first on Macro-F1; Gradient Boosting has the best log loss, Brier and RPS by small margins; Logistic Regression
  the best accuracy, balanced accuracy, kappa and MCC.
* **09** Gradient Boosting gave the actual results 9.2 log units more than the Random Forest (z = 1.86, not significant at 5%); Logistic
  Regression −0.4 (z = −0.05); all other models are clearly worse.
* **10** For Naive Bayes, PCA or calibration bring log loss down from 2.35 to 0.97-0.99 but Macro-F1 falls to 0.41 because draws disappear;
  removing the nine derived features gives 1.55 and 0.471. For the Random Forest, removing them gives 0.498 (+0.006, within the 0.01 rule).
* **11** All nine leagues are in every part of the split; train and test shares differ by at most 2.2 points (France, after Ligue 1 went to
  18 teams). Equal league weights: validation Macro-F1 0.4916 to 0.4919 (+0.0003); per league −0.007 to +0.006. Correlation between a
  league's training matches and its test Macro-F1: 0.03.
* **12** Train / validation / test: accuracy 0.492 / 0.506 / 0.489, Macro-F1 0.483 / 0.492 / 0.473. On the test seasons, always "home"
  scores 43.4% accuracy; the XI-gap rule 51.4% accuracy but Macro-F1 0.379 (it never predicts a draw). Bet365 favourite on the Stage 1
  seasons (22,592 matches with odds): 53.3% accuracy, Macro-F1 0.389, draws favoured in 0.14% of matches.
  Calibration (test): P(Home) averages 34.8% against 43.4% home wins, P(Draw) 32.9% against 25.6% draws, P(Away) 32.4% against 31.1%.
  When the highest of the three probabilities is 0.60 or more (15.0% of test matches) the pick is right 73.8% of the time; below 0.40
  (42.3% of matches) 34.8%.
