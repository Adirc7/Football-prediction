# Official dataset files: IT3091 Group 2026-DS-09

These four files are the official, verified set. Keep exactly these in the project folder.

| File | Rows x columns | What it is | Used by |
|---|---|---|---|
| `Stage1_PlayerAware_Football_2008_2016_Causal_v2.csv` | 25,979 x 62 | Engineered SQLite data, 2008/09-2015/16, 11 leagues | Build step, R script 01 |
| `Stage2_PlayerAware_Football_2016_2026_Causal_v2.csv` | 29,606 x 62 | Engineered Transfermarkt data, 2016/17-2025/26, 11 leagues | Build step, R script 01 |
| `European_Football_PlayerAware_Master_2008_2026.csv` | 55,585 x 62 | Stage 1 + Stage 2 stacked (11 leagues) | R script 01 (data journey) |
| `European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv` | 51,201 x 62 | Master without Switzerland (C1) and Poland (PL1). **The final dataset** | ML notebook, R scripts 02-04, pattern notebook |

How they fit together (checked row by row, 0 differences):
* Master = Stage 1 + Stage 2 (Causal_v2)
* Primary = Master without C1 and PL1

## Delete this file if you have it
`Stage2_PlayerAware_Football_2016_2026.csv` (without "Causal_v2" in the name). It is an older version: 28,422 of 29,606 rows have different strength values and some column names differ. None of the official files was built from it.

## The raw CSV folders are fine
`csv/stage_1/` and `csv/stage_2/` stay as they are. They are only needed to rebuild these files with `Data_Engineering_Build_Dataset.ipynb`.

## Check your copies (optional)
Compare the MD5 checksum of each file on your computer with this list. On Windows run `certutil -hashfile <file> MD5` in Command Prompt.
* `European_Football_PlayerAware_Master_2008_2026.csv`: 5189d36396819e7f817b4ecf6d3fb568
* `European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv`: f3cee88490552a994e4d4608fedcf938
* `Stage1_PlayerAware_Football_2008_2016_Causal_v2.csv`: 2e73c7123764e56118fefdcfe777faa1
* `Stage2_PlayerAware_Football_2016_2026_Causal_v2.csv`: f43d6ea88faeda869926f5b1ba48e6ab

The Primary file may show a different checksum if it was saved with Windows line endings; its content is identical (checked cell by cell).
