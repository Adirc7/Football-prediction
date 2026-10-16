"""Build extra pre-match features for the Stage 2 (Transfermarkt) matches: rest days, discipline and player form.

Every value uses only matches BEFORE the one being predicted (shift(1) before each rolling window), so nothing
from the match itself is used. Output: new_features_tm.csv (one row per Stage 2 match), read by script 02.

Run from this folder:  python 01_build_new_prematch_features.py --stage2-dir "<path to csv/stage_2>"
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ML_DIR = HERE.parent
ap = argparse.ArgumentParser()
ap.add_argument("--stage2-dir", default="csv/stage_2", help="raw Transfermarkt folder (games.csv, appearances.csv, game_lineups.csv)")
ap.add_argument("--out", default=str(HERE / "new_features_tm.csv"))
args = ap.parse_args()
U = Path(args.stage2_dir)

P = pd.read_csv(ML_DIR / "European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv", low_memory=False, parse_dates=["date"])
tm = P[P.data_source == "transfermarkt_stage2"][["match_uid", "game_id", "date", "home_team_id", "away_team_id"]]

# 1. Rest days and fixture congestion, counting every competition in games.csv
g = pd.read_csv(U / "games.csv", usecols=["game_id", "date", "home_club_id", "away_club_id"], parse_dates=["date"]).dropna()
cg = pd.concat([g[["game_id", "date", "home_club_id"]].set_axis(["game_id", "date", "club_id"], axis=1),
                g[["game_id", "date", "away_club_id"]].set_axis(["game_id", "date", "club_id"], axis=1)]).drop_duplicates(["game_id", "club_id"])
cg = cg.sort_values(["club_id", "date", "game_id"]).reset_index(drop=True)
cg["rest_days"] = cg.groupby("club_id").date.diff().dt.days.clip(upper=30)        # days since the club's previous match


def n14(s):  # number of matches in the 14 days BEFORE this one
    d = s.values.astype("datetime64[D]").astype(np.int64)
    return pd.Series(np.searchsorted(d, d, side="left") - np.searchsorted(d, d - 14, side="left"), index=s.index)


cg["matches_prev14"] = cg.groupby("club_id", group_keys=False).date.apply(n14)

# 2. Team discipline: cards per match over the club's previous 5 matches with appearance data
a = pd.read_csv(U / "appearances.csv", usecols=["game_id", "player_id", "player_club_id", "date", "yellow_cards", "red_cards",
                                                "goals", "assists", "minutes_played"], parse_dates=["date"])
tc = a.groupby(["game_id", "player_club_id", "date"])[["yellow_cards", "red_cards"]].sum().reset_index().rename(columns={"player_club_id": "club_id"})
tc = tc.sort_values(["club_id", "date", "game_id"])
gr = tc.groupby("club_id")
tc["yellow_L5"] = gr.yellow_cards.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
tc["red_L5"] = gr.red_cards.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
tc["red_prev_match"] = gr.red_cards.shift(1)                                         # possible suspension proxy

# 3. Player form: each starter's previous 5 appearances (any competition), summed over the starting XI
a = a.sort_values(["player_id", "date", "game_id"])
pg = a.groupby("player_id")
for c in ["goals", "assists"]:
    a[c + "_L5"] = pg[c].transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
a["minutes_played_L5"] = pg["minutes_played"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
st = []
for ch in pd.read_csv(U / "game_lineups.csv", usecols=["game_id", "player_id", "club_id", "type"], chunksize=500000):
    st.append(ch[(ch.type == "starting_lineup") & ch.game_id.isin(set(tm.game_id))])
st = pd.concat(st).merge(a[["game_id", "player_id", "goals_L5", "assists_L5", "minutes_played_L5"]], on=["game_id", "player_id"], how="left")
xi = st.groupby(["game_id", "club_id"]).agg(xi_goals_L5=("goals_L5", "sum"), xi_assists_L5=("assists_L5", "sum"),
                                            xi_minutes_L5=("minutes_played_L5", "mean"), xi_form_known=("goals_L5", "count")).reset_index()
xi.loc[xi.xi_form_known < 8, ["xi_goals_L5", "xi_assists_L5", "xi_minutes_L5"]] = np.nan   # same 8-starter rule as XI strength

# Assemble home and away columns for our matches
out = tm.copy()
for side in ("home", "away"):
    k = f"{side}_team_id"
    ren = lambda c: c if c == "game_id" else (k if c == "club_id" else f"{side}_{c}")
    out = out.merge(cg[["game_id", "club_id", "rest_days", "matches_prev14"]].rename(columns=ren), on=["game_id", k], how="left")
    out = out.merge(tc[["game_id", "club_id", "yellow_L5", "red_L5", "red_prev_match"]].rename(columns=ren), on=["game_id", k], how="left")
    out = out.merge(xi.drop(columns="xi_form_known").rename(columns=ren), on=["game_id", k], how="left")
out["rest_days_diff"] = out.home_rest_days - out.away_rest_days
out["xi_goals_L5_diff"] = out.home_xi_goals_L5 - out.away_xi_goals_L5
new = [c for c in out.columns if c not in tm.columns]
print("New features:", len(new))
print((out[new].isna().mean() * 100).round(1).rename("% missing").to_string())
out.to_csv(args.out, index=False)
print("Saved", args.out)
