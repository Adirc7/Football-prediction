# Build the final dataset from the raw CSV folders

**`Build_Master_Dataset.ipynb`** (notebook) and `Build_Master_Dataset.py` (same code as a script) turn the two raw CSV folders into the engineered Stage 1 and Stage 2 files, then into the
Master (11 leagues) and Primary (9 leagues) datasets.

```
csv/stage_1/  country, League, Match, Player Attributes, player, Team, Team Attributes, sqlite sequnce
csv/stage_2/  appearances, club_games, clubs, competitions, countries, game_events, game_lineups, games,
              national_teams, player_valuations, players, transfers
```
Only these files are read: Stage 1 = Match, League, country, Team, Player Attributes.
Stage 2 = games, competitions, game_lineups, player_valuations. The others are ignored.

## Run the notebook
Open `Build_Master_Dataset.ipynb`, edit the paths in Section 2, then Kernel > Restart & Run All.
`KEEP_ONLY_PRIMARY_LEAGUES = True` (default) uses the 9 primary leagues everywhere; set it to `False` for 11 leagues in the stage files and master.

## Run the script instead
```
pip install pandas numpy
python Build_Master_Dataset.py --stage1-dir "csv/stage_1" --stage2-dir "csv/stage_2" --output-dir output
```
Add `--existing-stage1 <your Stage1 csv> --existing-stage2 <your Stage2 csv>` to get a validation report
(`validation_stage1.csv`, `validation_stage2.csv`) comparing the rebuilt files with the engineered files you already have.

By default only the **primary** dataset is saved as the final file (set `SAVE_MASTER_FILE = True` / `--save-master` to also save the master).

Outputs in `output/`: Stage1_..._Causal_v2.csv, Stage2_..._Causal_v2.csv, European_Football_PlayerAware_Master_2008_2026.csv,
European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv.

## Self-test
```
python selftest_build.py --stage1 <engineered Stage1 csv> --stage2 <engineered Stage2 csv>
```

## What was validated, and how
Both stages were run on your real raw CSV files and compared with your existing engineered files
(`validation_stage1_real_raw_data.csv`, `validation_stage2_real_raw_data.csv`).

| | Stage 1 (25,979 matches) | Stage 2 (29,606 matches) |
|---|---|---|
| ids, dates, goals, target, league, teams | 100% | 100% |
| recent form (L5), starter counts, strength evidence counts, coverage | 100% | 100% |
| formation, lineup continuity, player_feature_ready | 100% | 100% |
| 0-100 strength columns | within 0.05 points on 99.9% of rows (mean abs. diff 0.0003) | 100% |
| mean age of ratings / valuations | exact on 99.4% | exact on 99.9% |
| columns identical (of 61) | 45 | 59 |

**Full pipeline, nine leagues:** the rebuilt primary dataset has 51,201 rows, the same 62 columns and the same target
counts as your existing primary file (H 23,020 / D 12,773 / A 15,408); 45 of 61 columns are identical, and the rest differ
only by tiny amounts (the Belgian defence-strength difference in Stage 1, a few mean-age values, one match with a
different starter count).

Rules found by calibrating against the real data
* Stage 1 roles from pitch Y: goalkeeper = 1, defence = 2-4, midfield = 5-8, attack = 9+; formation is the DEF-MID-ATT count of the same coordinates.
* Stage 2 roles from the lineup position text (a defensive midfielder is a midfielder).
* Percentile = 100 * (mid-rank - 0.5) / (n - 1) over earlier dates in the same league and role, issued only when the reference has at least 30 values. Stage 2 also uses 2012-2015 league lineups as a warm-up reference.
* Values must be at most 730 days old and dated strictly before the match.
* xi_strength needs at least 8 starters with a strength value, top-3 at least 3.
* Lineup continuity compares with the team's latest earlier same-season match that has recorded starters and needs two complete XIs of 11.

## Parameters you may need to tune (top of the script)
none are needed for your data; all constants are now verified against it.
Run with `--existing-stage1/2` and compare: form, goals, targets and starter counts should match ~100%; strength columns
should show a Spearman correlation close to 1 with your existing values.
