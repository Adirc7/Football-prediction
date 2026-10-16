# R analysis: descriptive analysis and statistical inference

Run the scripts in RStudio in this order. Put the CSV files in the same folder as the scripts, or set that folder with Session > Set Working Directory.

| Script | Part | What it answers | Data used |
|---|---|---|---|
| `01_Data_Journey.R` | Descriptive | How the dataset was made: raw tables, then Stage 1 and Stage 2, then Master, then Primary. Rows removed and why, how the two sources join in time, coverage and missing values by stage, and the current state of the final file. | All seasons. It describes the data only. |
| `02_Descriptive_Analysis.R` | Descriptive | Distributions, outcome shares, leagues, missing values, skewness, correlations, XI strength and form. | All seasons for counts. Training seasons for anything that relates a feature to the outcome. |
| `03_Statistical_Inference.R` | Inference | Does each feature differ between H, D and A? Uses Levene, Welch ANOVA, Kruskal-Wallis with effect sizes, Wilcoxon post-hoc tests, chi-square for league, and correlation tests, with a Holm correction. | Training seasons only (2008/09 to 2022/23). |
| `04_Time_and_Model_Inference.R` | Inference | Is the data time dependent? Multicollinearity (VIF). A formal test of RQ2: does player information improve on team form? Uses nested multinomial logistic regressions with likelihood-ratio tests. | Training seasons only. |

Files needed:
* Scripts 02, 03 and 04: `European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv`
* Script 01: also `Stage1_..._Causal_v2.csv`, `Stage2_..._Causal_v2.csv` and `European_Football_PlayerAware_Master_2008_2026.csv`
* Script 01, optional: the raw `csv/stage_1` and `csv/stage_2` folders, for raw row counts

Packages: `tidyverse`, `data.table`, `scales`, `corrplot`, `car`, `nnet`. Install them once with `install.packages(...)`.

`results_from_verification_run/` holds the tables and figures from running all four scripts on the real files. Scripts 02 and 03 draw their figures in the RStudio Plots pane.

## Key results (all from the verification run)

**Dataset journey**
* Raw files: 25,979 SQLite matches and 88,958 Transfermarkt games.
* Engineered stages: Stage 1 has 25,979 matches and Stage 2 has 29,606.
* Master: 55,585 matches. Primary: 51,201 matches, after removing 4,384 matches (7.9%) from Switzerland and Poland.
* Player data is usable for 93.5% of matches. The largest gap is 7.4%, in lineup continuity.

**Time dependence (training seasons)**
* Outcome shares differ between seasons, but only a little: chi-square p < 0.001 with Cramér's V = 0.034, a very small effect.
* Home wins in the COVID season 2020/21 were 40.2%, against 45.7% in the other seasons (p < 0.001).
* There is a small long-term decline in home wins (odds ratio 0.993 per season, Holm p = 0.014).
* The effect of XI strength does **not** differ between Stage 1 and Stage 2 (LR test p = 0.44). Its effect size is stable across seasons (epsilon-squared 0.11 to 0.18).

**Multicollinearity**
* Four exact identities: for example, goal difference = goals for − goals against, and `xi_strength_diff` = home − away.
* Among the other 26 features, 14 have VIF above 10. The highest is XI strength at about 196, because it overlaps with the attack, midfield and defence strengths.
* This doesn't affect Random Forest predictions. It does mean importance should be read by feature group, not one feature at a time.

**RQ2: nested multinomial logistic regressions (37,347 training matches)**

| Model | McFadden R² | LR test against the previous model |
|---|---|---|
| League only | 0.001 | |
| + recent form | 0.039 | p < 2.2e-16 |
| + starting-XI quality | 0.080 | p < 2.2e-16 |
| + positional matchups | 0.080 | Holm p = 0.001, a tiny gain |
| + lineup continuity | 0.080 | p = 0.38, not significant |

One standard deviation more XI-strength advantage for the home team multiplies the odds of a home win against a draw by 1.53 (95% CI 1.48 to 1.58).
