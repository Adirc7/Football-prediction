# Football-prediction
Check the win probability before kickoff.

## IT3091 Machine Learning · Group 2026-DS-09 · Final submission

**Project:** European football match outcome prediction (home win / draw / away win) after the starting line-ups are announced and before kickoff.
**Final result:** Random Forest on 38 pre-match features. On the 2024/25–2025/26 test seasons (5,604 matches): Macro-F1 0.473 (95% CI 0.460–0.486), accuracy 48.9%, against 0.202 / 43.4% for the baseline.

| Member | Student ID | Part |
|---|---|---|
| Kumara D.G.K.R.D. | IT24103972 | Problem framing, data sources and integration |
| Jayamuni J T S J | IT24300329 | EDA and data quality |
| Jayawardana D. I. A. | IT24103756 | Feature engineering and preprocessing |
| Bandara H. M. T. A. | IT24103675 | Modelling, evaluation and interpretation |

### Contents (one folder per required item)

| Required item | Folder | What is inside |
|---|---|---|
| Final report | `1_Final_Report/` | The report in Word and PDF |
| Final notebook / code | `2_Notebook_and_Code/` | `ML_Pipeline/`: the main machine-learning notebook (executed), its result tables, figures, saved final model and the supporting experiment scripts. Also the dataset build notebook, the cross-source check, the feature-engineering code, the R scripts for the EDA and statistical tests, and the pattern-exploration notebook |
| Dataset / data dictionary | `3_Dataset_and_Data_Dictionary/` | The final dataset (`European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv`, 51,201 matches × 62 columns, MD5 `f3cee88490552a994e4d4608fedcf938`), `Data_Dictionary.xlsx` (all 62 columns) and the checksums |
| Decision logs | `4_Decision_Logs/` | `Decision_Log.xlsx` (20 decisions), `Data_Issues_Register.xlsx` (26 issues), `EDA_Insight_Log.xlsx` (20 insights), `Preprocessing_Feature_Log.xlsx` (features, preprocessing steps, leakage checks) |
| Model / method comparison | `5_Model_Comparison/` | `Model_Choice_Justification.pdf`, model by model. The full comparison is in report Sections 17–18; the result tables are in `2_Notebook_and_Code/ML_Pipeline/ml_outputs/` |
| Personal Learning Journey (one A4 page each) | `6_Personal_Learning_Journey/` | Four PDFs, one per member |
| 3-minute demo video | `7_Video/` | `YouTube_Link.txt` |

### Running the code
* Every notebook is saved with its outputs, so it can be read without re-running.
* To re-run the main notebook: copy the dataset CSV from `3_Dataset_and_Data_Dictionary/` into `2_Notebook_and_Code/ML_Pipeline/`, run `pip install -r requirements.txt`, then *Restart & Run All* (about 25–30 minutes). The supporting scripts in `ML_Pipeline/extra_experiments/` run from the `ML_Pipeline` folder in the same way (see its README).
* The R scripts 02–04 need the same CSV in their folder. The dataset build notebook, the cross-source check, R script 01 and parts of the pattern notebook also need the raw Kaggle and Transfermarkt files, which are not included because of their size.

### Folder names used in the report
The report (Appendix I) describes the evidence by member folder. In this package:

| Report reference | Location here |
|---|---|
| folder 1: Data_Engineering, Cross_Source_Check | `2_Notebook_and_Code/Data_Engineering/`, `2_Notebook_and_Code/Cross_Source_Check/` |
| folder 1: Decision_Log.xlsx, Data_Issues_Register.xlsx | `4_Decision_Logs/` |
| folder 2: R scripts 01–04, Pattern_Exploration.ipynb | `2_Notebook_and_Code/R_Scripts/`, `2_Notebook_and_Code/Pattern_Exploration/` |
| folder 2: EDA_Insight_Log.xlsx | `4_Decision_Logs/` |
| folder 3: Build_Pipeline | `2_Notebook_and_Code/Build_Pipeline/` |
| folder 3: Preprocessing_Feature_Log.xlsx | `4_Decision_Logs/` |
| folder 4: ML_Pipeline | `2_Notebook_and_Code/ML_Pipeline/` |
| folder 4: Model_Choice_Justification | `5_Model_Comparison/` |
| the final dataset (folders 1, 2 and 4) | `3_Dataset_and_Data_Dictionary/` |
