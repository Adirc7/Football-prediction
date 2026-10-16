"""Alternative Stage 2 strength features: each player value is ranked only against the SAME season.

Used by notebook Section 25 (variant V5) to test whether transfer-fee inflation over the years distorts the
official ranking (which compares with all earlier dates). Output: ml_outputs/stage2_strength_same_season_ranking.csv.

Run from this folder:  python 04_same_season_ranking.py --stage2-dir "<path to csv/stage_2>"
"""
import argparse, sys
from bisect import bisect_left, bisect_right, insort
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ML_DIR = HERE.parent
ap = argparse.ArgumentParser()
ap.add_argument("--stage2-dir", default="csv/stage_2")
ap.add_argument("--build-dir", default=str(ML_DIR.parent / "Build_Pipeline"))
ap.add_argument("--out-dir", default=str(ML_DIR / "ml_outputs"))
args = ap.parse_args()
sys.path.insert(0, args.build_dir)
import Build_Master_Dataset as B


def causal_percentile_same_season(starters, value_col="raw_value"):
    # identical to B.causal_percentile, but the reference group is (league, role, SEASON): it restarts every July
    ordered = starters.sort_values("date", kind="stable")
    d = pd.to_datetime(ordered["date"]); season = (d.dt.year - (d.dt.month < 7)).to_numpy()
    dates = ordered["date"].to_numpy(); leagues = ordered["league_code"].to_numpy(dtype=object)
    roles = ordered["role"].to_numpy(dtype=object); values = ordered[value_col].to_numpy(dtype=float)
    result = np.full(len(ordered), np.nan); ref = {}; n = len(ordered); i = 0
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        for k in range(i, j):                                   # rank against values from EARLIER dates only
            if np.isnan(values[k]) or B._is_missing(roles[k]):
                continue
            r = ref.get((leagues[k], roles[k], season[k]))
            if r is not None and len(r) >= B.MIN_REFERENCE_N:
                lo, hi = bisect_left(r, values[k]), bisect_right(r, values[k])
                result[k] = min(100.0, max(0.0, 100.0 * ((lo + hi) / 2.0 - 0.5) / (len(r) - 1)))
        for k in range(i, j):                                   # then add today's values to the reference
            if np.isnan(values[k]) or B._is_missing(roles[k]):
                continue
            insort(ref.setdefault((leagues[k], roles[k], season[k]), []), values[k])
        i = j
    return pd.Series(result, index=ordered.index).reindex(starters.index)


B.causal_percentile = causal_percentile_same_season
s2 = B.build_stage2(Path(args.stage2_dir))
cols = ["match_uid"] + [c for c in s2.columns if any(k in c for k in ["strength", "top3", "known_n", "coverage", "_vs_"])] + ["player_feature_ready"]
out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
s2[cols].to_csv(out / "stage2_strength_same_season_ranking.csv", index=False)
print(s2.groupby("season_start").home_xi_strength.mean().round(1).to_dict())
print("Saved", out / "stage2_strength_same_season_ranking.csv")
