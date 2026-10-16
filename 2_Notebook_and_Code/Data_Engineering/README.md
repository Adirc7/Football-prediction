# Data Engineering: IT3091 Group 2026-DS-09

| File | What it is |
|---|---|
| `Data_Engineering_Build_Dataset.ipynb` | Step-by-step notebook: raw CSV files -> Stage 1 -> Stage 2 -> Master -> Primary, then independent verification. Already run on the real raw files, so all outputs are visible. |
| `European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv` | The final dataset used by the ML pipeline (51,201 matches, 62 columns, 9 leagues). In this submission it is in `3_Dataset_and_Data_Dictionary/`. |
| `verification_checks.csv` | The 14 checks from Part E (all PASS). |
| `comparison_with_ML_dataset.csv` | Column-by-column comparison of the rebuilt Primary file with the final dataset. |

## Run it in Google Colab
1. Upload your `csv` folder (containing `stage_1` and `stage_2`) to Google Drive, for example to `My Drive/Machine Learning/Assignment/csv`.
2. Open the notebook in Colab (File > Upload notebook).
3. In the first code cell, set `DRIVE_FOLDER` to the folder that contains `csv` (for example `/content/drive/MyDrive/Machine Learning/Assignment`).
4. Optional: put the final dataset CSV in the same folder and set `EXISTING_PRIMARY = str(BASE / "European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv")` to run the comparison in Part E5.
5. Runtime > Run all. It takes about 2 minutes on a normal computer, and may take longer in Colab because the files are read from Google Drive. Results are written to `output/` in the same folder.

## Run it in Jupyter on your computer
Put the notebook in your `Assignment` folder (next to `csv`) and run all cells. Only `pandas` and `numpy` are needed.

## Raw files used
* Stage 1: `Match`, `League`, `country`, `Team`, `Player Attributes`
* Stage 2: `games`, `competitions`, `game_lineups`, `player_valuations`

The other raw files are not needed. `game_events` and `appearances` describe what happened during matches, so using them would leak the result.

## Verification result (Part E)
* Recent form recomputed with a simple loop on 1,000 random team-matches: identical.
* Player percentiles recomputed by brute force on 300 random starters per stage: identical.
* All 1.37 million ratings and valuations are dated strictly before the match day and are at most 730 days old.
* Target matches the goals in all 51,201 rows. There are no duplicates, all strength values are within 0 to 100, and the stages do not overlap.
* Compared with the final dataset: same 51,201 matches. Targets, teams, dates, form and lineup continuity are identical in every row. 45 of 61 columns are identical in every row. The other columns differ only in player strength for some Belgian Stage 1 matches (median 0.003 points on the 0 to 100 scale), in a few mean-age values, and in one match whose starter count differs by one.
