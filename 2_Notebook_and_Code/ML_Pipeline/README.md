# IT3091 Machine Learning Pipeline: European Football Match Outcome Prediction

`Machine_Learning_Pipeline.ipynb` implements the full ML stage on the final engineered dataset
(three-class target H / D / A, prediction made after the starting XI is known and before kickoff).

## 1. Dataset required
`European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv` (51,201 matches, 62 columns).

In this submission the file is in `3_Dataset_and_Data_Dictionary/` (one copy only); copy it next to the notebook before re-running. It was produced from `European_Football_PlayerAware_Master_2008_2026.csv`
by removing the two leagues with partial Stage 2 coverage (league codes `C1` Switzerland and `PL1` Poland). No values
were changed. Row count, target counts (H 23,020 / D 12,773 / A 15,408) and date range (2008-08-09 to 2026-05-24)
match the project brief.

## 2. Where to put the CSV
In the **same folder as the notebook** (or in a `data/` sub-folder). The notebook only reads it and never writes to it.

## 3. Install packages
```
pip install -r requirements.txt
```

## 4. Run the notebook
```
jupyter notebook Machine_Learning_Pipeline.ipynb
```
Choose *Kernel > Restart & Run All*. Or run headless:
```
jupyter nbconvert --to notebook --execute --inplace Machine_Learning_Pipeline.ipynb
```
Expect roughly 25 to 30 minutes on a single CPU core (the FS1 to FS4 ablation and the temporal grid searches dominate).
The test set is scored once per session; a guard stops the cell from running twice, so use *Restart & Run All* to repeat.

## 5. Outputs (created in `ml_outputs/`)
| File | Content |
|---|---|
| `ablation_validation_results.csv` | FS1 to FS4 x all models, validation metrics |
| `tuned_validation_results.csv` | Temporal-CV tuned models and their validation metrics |
| `candidate_ranking_validation.csv` | Every candidate ranked by validation Macro-F1 |
| `final_model_selection.csv` | The frozen configuration |
| `final_test_metrics.csv` | Single test evaluation, incl. class-wise metrics and Macro-F1 bootstrap CI |
| `final_test_predictions.csv` | match_uid, date, teams, actual, predicted, P(Home), P(Draw), P(Away) |
| `feature_importance.csv` | Permutation importance (columns and feature groups) |
| `confusion_matrix.csv` | Test confusion matrix |
| `final_feature_list.csv` | Columns used by the frozen model |
| `final_model.joblib`, `final_preprocessor.joblib` | Fitted pipeline and preprocessor |
| `figures/*.png` | All charts |

## Design notes
* **Split:** train `season_start <= 2022`, validation `== 2023`, test `>= 2024`.
* **No leakage:** `audit_home_goals`, `audit_away_goals`, `target_result`, identifiers and metadata are excluded; the
  notebook asserts this. Preprocessing lives inside each `Pipeline`, so it is fitted on training data only.
* **Tuning:** `GridSearchCV` with `TimeSeriesSplit(3)` inside the training period, `scoring="f1_macro"`.
* **Selection:** validation Macro-F1 only; the configuration is frozen before the test set is scored.
* **Models:** Dummy (prior), Logistic Regression, Gaussian NB, kNN, Decision Tree, Random Forest, Gradient Boosting.
  No SMOTE is used; class imbalance is handled by comparing `class_weight=None` with `"balanced"`.
* **Reading Log Loss and Macro-F1 together:** a class-weighted model can score higher on Macro-F1 while producing
  probabilities with a worse Log Loss. Sections 15, 19 and 23 of the notebook report both.

## Section 26: additional models (added later)
Five further model families (MLP neural network, HistGradientBoosting, XGBoost, linear and RBF-kernel SVM, ordinal logistic regression) are compared with the frozen Random Forest on the validation season only. Results: `ml_outputs/additional_models_validation.csv`. None beat the Random Forest by more than 0.01, so the final model is unchanged. This section needs `xgboost` and `statsmodels` (in `requirements.txt`) and was run with scikit-learn 1.9, so the Random Forest is retrained inside the section as the reference.

## Section 27: Poisson goal model (added later)
The classic statistical approach from the football literature: Poisson regression predicts the expected goals of each team from the
same 38 features, the two goal counts give the chance of every scoreline, and the scorelines are added up into P(Home), P(Draw) and
P(Away). Compared with the frozen Random Forest on the validation season only. Validation Macro-F1 0.405 (most probable result; it never
predicts a draw) or 0.453 (prior-corrected decision), against 0.492 for the Random Forest, so the final model is unchanged; its
probabilities are slightly better (log loss 0.969 vs 0.983). The section runs `extra_experiments/06_poisson_goal_model.py` unchanged and
was executed with scikit-learn 1.8, the same version as Sections 1-24. Results: `ml_outputs/poisson_validation.csv`,
`poisson_goal_checks.csv`, `poisson_coefficients.csv`, `figures/poisson_scoreline_grid.png`.

## Extra experiments
`extra_experiments/` holds twelve scripts whose results the report quotes (feature tests, AdaBoost, the Poisson model, the F1 types and
other selection measures, the feature-overlap check, the league-balance check, and the split, benchmark and calibration tables). See
`extra_experiments/README.md` for what each answers, how to run it and the results.
