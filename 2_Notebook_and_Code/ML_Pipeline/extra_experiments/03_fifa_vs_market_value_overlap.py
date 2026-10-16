"""Do the two player-strength scales agree? (FIFA rating in Stage 1 vs market value in Stage 2)

Stage 2 is rebuilt for 2012-2015 (the years the Transfermarkt files also cover) and matched one-to-one with the
Stage 1 matches of the same league, date and score. Prints the correlation of each strength feature between the two
versions (report: r = 0.94 for xi_strength_diff).

Run from this folder:  python 03_fifa_vs_market_value_overlap.py --stage2-dir "<path to csv/stage_2>"
"""
import argparse, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--stage2-dir", default="csv/stage_2")
ap.add_argument("--stage1-file", default=str(PROJECT / "Official_Dataset_Files" / "Stage1_PlayerAware_Football_2008_2016_Causal_v2.csv"))
ap.add_argument("--build-dir", default=str(PROJECT / "Build_Pipeline"), help="folder containing Build_Master_Dataset.py")
args = ap.parse_args()
sys.path.insert(0, args.build_dir)
import Build_Master_Dataset as B

s2 = B.build_stage2(Path(args.stage2_dir), first_season=2012)
s2["date"] = pd.to_datetime(s2.date)
tm = s2[s2.season_start.between(2012, 2015)].copy()
s1 = pd.read_csv(args.stage1_file, low_memory=False)
s1["date"] = pd.to_datetime(s1.date)
s1 = s1[s1.season_start.between(2012, 2015) & s1.league_code.isin(B.PRIMARY_LEAGUES)]
k = ["league_code", "date", "audit_home_goals", "audit_away_goals"]
for d in (s1, tm):
    d["dup"] = d.duplicated(k, keep=False)               # keep only keys that identify one match in each source
m = s1[~s1.dup].merge(tm[~tm.dup], on=k, suffixes=("_fifa", "_mv"))
print("Stage 1 matches 2012-15:", len(s1), "| matched one-to-one:", len(m))
for c in ["home_xi_strength", "away_xi_strength", "xi_strength_diff", "home_attack_strength", "home_defence_strength", "home_gk_strength"]:
    a, b = m[c + "_fifa"], m[c + "_mv"]; ok = a.notna() & b.notna()
    print(f"{c:24s} n={ok.sum():6d} pearson={a[ok].corr(b[ok]):.3f} spearman={a[ok].corr(b[ok], method='spearman'):.3f}")
