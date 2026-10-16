#!/usr/bin/env python3
# ============================================================
# IT3091 - FOOTBALL ANALYTICS
# BUILD THE FINAL DATASET FROM THE RAW CSV FILES
#
#   stage_1/  (European Soccer SQLite tables exported to CSV)
#   stage_2/  (Transfermarkt CSV bundle)
#        |                       |
#        v                       v
#   Stage 1 builder         Stage 2 builder
#   (source-specific feature construction)
#        |                       |
#        v                       v
#   common 62-column analytical schema
#                  |
#                  v
#   row-wise concatenation  ->  MASTER (11 leagues)
#                  |
#                  v
#   keep nine leagues       ->  PRIMARY (51,201 matches)
#
# HOW TO RUN
#   python Build_Master_Dataset.py --stage1-dir "csv/stage_1" --stage2-dir "csv/stage_2"
#
# OPTIONAL VALIDATION against engineered files you already have:
#   python Build_Master_Dataset.py --stage1-dir ... --stage2-dir ... \
#       --existing-stage1 Stage1_..._Causal_v2.csv --existing-stage2 Stage2_..._Causal_v2.csv
#
# IMPORTANT HONESTY NOTE
#   Everything marked [VERIFIED] was checked against your engineered CSVs.
#   Everything marked [ASSUMPTION] is a design choice that follows the project
#   documentation but could not be checked without the raw files; the validation
#   report (--existing-stage1/2) tells you how closely the rebuilt data agree
#   with your existing files, and the constants in the CONFIG block can be tuned.
# ============================================================

from __future__ import annotations

import argparse
import re
import sys
from bisect import bisect_left, bisect_right, insort
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
FORM_WINDOW = 5                 # [VERIFIED] last 5 earlier same-season league matches
SAVE_MASTER_FILE = False           # True: also write the 11/9-league Master CSV; False: write only the PRIMARY dataset
KEEP_ONLY_PRIMARY_LEAGUES = True   # True: use the 9 primary leagues everywhere (drops Switzerland C1 and Poland PL1)
MAX_VALUE_AGE_DAYS = 730        # freshness limit for rating / valuation (observed max ages in your files: 702 / 509)
MIN_REFERENCE_N = 30            # [VERIFIED] minimum earlier player-appearances before a percentile is issued
MIN_COVERAGE_READY = 0.70       # [VERIFIED range] ready=1 needs both XIs complete and coverage >= ~0.70 (8/11 ready, 7/11 not)
ROUND_DECIMALS = 4              # [VERIFIED] the engineered files are rounded to 4 decimals

# Stage 1 pitch-coordinate bands used to assign a role to each starter.
# [VERIFIED on the real Match table: these bands reproduce 100% of the existing formation labels]
# Y = 1 goalkeeper | Y = 2-4 defence | Y = 5-8 midfield | Y >= 9 attack
Y_GK_MAX = 1
Y_DEF_MAX = 4
Y_MID_MAX = 8                   # Y above this value -> attacker
FULL_XI = 11                    # an XI counts as "complete" when 11 starters are identified
REFERENCE_FIRST_SEASON_STAGE2 = 2012   # [VERIFIED] Stage 2 percentile reference also uses 2012-2015 league lineups (warm-up)
MIN_KNOWN_FOR_XI = 8            # [VERIFIED in both engineered files] xi_strength needs >= 8 starters with a strength value
MIN_KNOWN_FOR_TOP3 = 3          # [VERIFIED] top3_strength needs >= 3 starters with a strength value

STAGE1_LEAGUE_CODES = {          # [VERIFIED] league name -> code used in the engineered Stage 1 file
    "Belgium Jupiler League": "BE1",
    "England Premier League": "GB1",
    "France Ligue 1": "FR1",
    "Germany 1. Bundesliga": "L1",
    "Italy Serie A": "IT1",
    "Netherlands Eredivisie": "NL1",
    "Poland Ekstraklasa": "PL1",
    "Portugal Liga ZON Sagres": "PO1",
    "Scotland Premier League": "SC1",
    "Spain LIGA BBVA": "ES1",
    "Switzerland Super League": "C1",
}
STAGE2_LEAGUE_CODES = ["BE1", "ES1", "FR1", "GB1", "IT1", "L1", "NL1", "PL1", "PO1", "SC1", "C1"]
PRIMARY_LEAGUES = ["BE1", "ES1", "FR1", "GB1", "IT1", "L1", "NL1", "PO1", "SC1"]

STAGE1_SOURCE = "european_soccer_sqlite_stage1"
STAGE1_METHOD = "fifa_overall_rating_causal_prior_league_role_percentile"
STAGE2_SOURCE = "transfermarkt_stage2"
STAGE2_METHOD = "market_value_log_causal_prior_league_role_percentile"

STAGE1_FILE = "Stage1_PlayerAware_Football_2008_2016_Causal_v2.csv"
STAGE2_FILE = "Stage2_PlayerAware_Football_2016_2026_Causal_v2.csv"
MASTER_FILE = "European_Football_PlayerAware_Master_2008_2026.csv"
PRIMARY_FILE = "European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv"

# The common analytical schema (identical to your master-creation script).
COMMON_COLUMNS = [
    "match_uid", "data_source", "strength_method", "game_id", "date", "season_start", "season",
    "league_code", "league_name", "country", "league_coverage_flag",
    "home_team_id", "away_team_id", "home_team", "away_team",
    "home_formation", "away_formation",
    "home_form_n_L5", "away_form_n_L5",
    "home_points_per_game_L5", "away_points_per_game_L5",
    "home_win_rate_L5", "away_win_rate_L5",
    "home_goals_for_avg_L5", "away_goals_for_avg_L5",
    "home_goals_against_avg_L5", "away_goals_against_avg_L5",
    "home_goal_diff_avg_L5", "away_goal_diff_avg_L5",
    "home_starter_count", "away_starter_count",
    "home_strength_known_n", "away_strength_known_n",
    "home_strength_coverage", "away_strength_coverage",
    "home_strength_age_mean_days", "away_strength_age_mean_days",
    "home_xi_strength", "away_xi_strength",
    "home_attack_strength", "away_attack_strength",
    "home_midfield_strength", "away_midfield_strength",
    "home_defence_strength", "away_defence_strength",
    "home_gk_strength", "away_gk_strength",
    "home_top3_strength", "away_top3_strength",
    "home_top3_gap", "away_top3_gap",
    "home_lineup_continuity", "away_lineup_continuity",
    "xi_strength_diff",
    "home_attack_vs_away_defence", "away_attack_vs_home_defence",
    "midfield_strength_diff", "gk_strength_diff",
    "player_feature_ready",
    "audit_home_goals", "audit_away_goals", "target_result",
]


# ------------------------------------------------------------
# GENERIC HELPERS
# ------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def find_csv(folder: Path, table_name: str) -> Path:
    """Find '<table_name>.csv' ignoring case, spaces and underscores (e.g. 'Player Attributes')."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder.resolve()}")
    wanted = _norm(table_name)
    for p in folder.iterdir():
        if p.suffix.lower() == ".csv" and _norm(p.stem) == wanted:
            return p
    raise FileNotFoundError(f"No CSV named like '{table_name}' in {folder.resolve()}. "
                            f"Found: {sorted(x.name for x in folder.glob('*.csv'))}")


def read_columns(path: Path, needed: list[str], **kwargs) -> pd.DataFrame:
    """Read only the needed columns and fail with a clear message if one is absent."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in needed if c not in header]
    if missing:
        raise KeyError(f"{path.name} is missing columns {missing}. Available: {header}")
    return pd.read_csv(path, usecols=needed, **kwargs)


def to_ns(series: pd.Series) -> pd.Series:
    """Datetime with nanosecond resolution (merge_asof needs identical dtypes)."""
    return pd.to_datetime(series, errors="coerce").astype("datetime64[ns]")


def season_label_short(start_year: int) -> str:
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def result_from_goals(home: pd.Series, away: pd.Series) -> pd.Series:
    return pd.Series(np.where(home > away, "H", np.where(home == away, "D", "A")), index=home.index)


def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and x != x) or x is pd.NA


# ------------------------------------------------------------
# FEATURE BLOCK 1: RECENT FORM (L5)                     [VERIFIED]
# ------------------------------------------------------------
def add_form_features(matches: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame:
    """
    For every team and season (league matches only), use the previous `window` matches,
    EXCLUDING the current one. Reset every season.
    Needs: game_id, date, season_start, home_team_id, away_team_id, home_goals, away_goals.
    Returns one row per game with home_/away_ form columns.
    """
    home = matches[["game_id", "date", "season_start", "home_team_id", "home_goals", "away_goals"]].rename(
        columns={"home_team_id": "team_id", "home_goals": "gf", "away_goals": "ga"})
    home["side"] = "home"
    away = matches[["game_id", "date", "season_start", "away_team_id", "away_goals", "home_goals"]].rename(
        columns={"away_team_id": "team_id", "away_goals": "gf", "home_goals": "ga"})
    away["side"] = "away"

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team_id", "season_start", "date", "game_id"]).reset_index(drop=True)
    long["points"] = np.where(long["gf"] > long["ga"], 3, np.where(long["gf"] == long["ga"], 1, 0))
    long["win"] = (long["gf"] > long["ga"]).astype(int)
    long["gd"] = long["gf"] - long["ga"]

    grouped = long.groupby(["team_id", "season_start"], sort=False)

    def prior_mean(col: str) -> pd.Series:
        return grouped[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())

    long["points_per_game_L5"] = prior_mean("points")
    long["win_rate_L5"] = prior_mean("win")
    long["goals_for_avg_L5"] = prior_mean("gf")
    long["goals_against_avg_L5"] = prior_mean("ga")
    long["goal_diff_avg_L5"] = prior_mean("gd")
    long["form_n_L5"] = (grouped["points"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).count())
                         .fillna(0).astype(int))

    form_cols = ["form_n_L5", "points_per_game_L5", "win_rate_L5", "goals_for_avg_L5",
                 "goals_against_avg_L5", "goal_diff_avg_L5"]
    out = matches[["game_id"]].copy()
    for side in ("home", "away"):
        part = long.loc[long["side"] == side, ["game_id"] + form_cols].copy()
        part.columns = ["game_id"] + [f"{side}_{c}" for c in form_cols]
        out = out.merge(part, on="game_id", how="left")
    for c in out.columns:
        if c != "game_id" and not c.endswith("form_n_L5"):
            out[c] = out[c].round(ROUND_DECIMALS)
    return out


# ------------------------------------------------------------
# FEATURE BLOCK 2: LINEUP CONTINUITY      [VERIFIED 100% on the real Stage 1 Match table]
# ------------------------------------------------------------
def add_lineup_continuity(matches: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    """
    Share of the current XI that also started the team's previous same-season league match.

    Rule (reproduces the existing Stage 1 file on every row):
      * "previous match" = the team's latest EARLIER match in the same season that has any recorded starters
        (matches with no lineup data at all are skipped);
      * the value exists only when the current XI AND that previous XI both have 11 identified starters;
      * value = |current XI intersect previous XI| / 11;  otherwise NaN.
    Needs matches: game_id, date, season_start, home_team_id, away_team_id
          starters: game_id, side, player_id
    """
    lineups = starters.groupby(["game_id", "side"])["player_id"].agg(frozenset)

    rows = []
    for side in ("home", "away"):
        part = matches[["game_id", "date", "season_start", f"{side}_team_id"]].rename(
            columns={f"{side}_team_id": "team_id"})
        part["side"] = side
        rows.append(part)
    long = pd.concat(rows, ignore_index=True)
    long["lineup"] = [lineups.get((g, s), frozenset()) for g, s in zip(long["game_id"], long["side"])]
    long = long.sort_values(["team_id", "season_start", "date", "game_id"]).reset_index(drop=True)

    continuity = np.full(len(long), np.nan)
    pos = 0
    for _, group in long.groupby(["team_id", "season_start"], sort=False):
        last_with_players = None
        for lineup in group["lineup"]:
            if (last_with_players is not None and len(lineup) == FULL_XI and len(last_with_players) == FULL_XI):
                continuity[pos] = len(lineup & last_with_players) / FULL_XI
            if len(lineup) > 0:
                last_with_players = lineup
            pos += 1
    long["continuity"] = continuity

    out = matches[["game_id"]].copy()
    for side in ("home", "away"):
        part = long.loc[long["side"] == side, ["game_id", "continuity"]].rename(
            columns={"continuity": f"{side}_lineup_continuity"})
        out = out.merge(part, on="game_id", how="left")
        out[f"{side}_lineup_continuity"] = out[f"{side}_lineup_continuity"].round(ROUND_DECIMALS)
    return out


# ------------------------------------------------------------
# FEATURE BLOCK 3: PLAYER STRENGTH (shared by both stages)
# ------------------------------------------------------------
def attach_latest_value(starters: pd.DataFrame, values: pd.DataFrame, id_col: str, value_date_col: str,
                        value_col: str, max_age_days: int = MAX_VALUE_AGE_DAYS) -> pd.DataFrame:
    """
    For each starter, attach the LATEST value dated STRICTLY BEFORE the match date
    (and at most `max_age_days` old). This is the causal rule: no future information.
    """
    left = starters.copy()
    left["date"] = to_ns(left["date"])
    left[id_col] = left[id_col].astype("int64")
    right = values[[id_col, value_date_col, value_col]].dropna().copy()
    right[value_date_col] = to_ns(right[value_date_col])
    right[id_col] = right[id_col].astype("int64")
    right = right.sort_values(value_date_col, kind="stable")
    left = left.sort_values("date", kind="stable")
    merged = pd.merge_asof(left, right, left_on="date", right_on=value_date_col, by=id_col,
                           direction="backward", allow_exact_matches=False,
                           tolerance=pd.Timedelta(days=max_age_days))
    merged["value_age_days"] = (merged["date"] - merged[value_date_col]).dt.days.astype(float)
    return merged.drop(columns=[value_date_col])


def causal_percentile(starters: pd.DataFrame, value_col: str = "raw_value") -> pd.Series:
    """
    [VERIFIED on the real Stage 1 data: reproduces every existing goalkeeper-strength value]
    Convert a raw player value (FIFA rating or log market value) to a 0-100 percentile within
    (league, role), using ONLY starters from STRICTLY EARLIER match dates as the reference.

    * Matches on the same date are all scored first, THEN added to the reference set.
    * A percentile is issued only when the reference holds at least MIN_REFERENCE_N = 30 values
      (verified: the existing file has values at 30 and none at 29 or fewer).
    * With `below` = reference values < v, `equal` = reference values == v and n = reference size:
          mid_rank   = below + equal / 2
          percentile = 100 * (mid_rank - 0.5) / (n - 1)      clipped to [0, 100]
    """
    ordered = starters.sort_values("date", kind="stable")
    dates = ordered["date"].to_numpy()
    leagues = ordered["league_code"].to_numpy(dtype=object)
    roles = ordered["role"].to_numpy(dtype=object)
    values = ordered[value_col].to_numpy(dtype=float)

    result = np.full(len(ordered), np.nan)
    reference: dict[tuple, list] = {}
    n, i = len(ordered), 0
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        for k in range(i, j):                                   # 1) score everything on this date
            if np.isnan(values[k]) or _is_missing(roles[k]):
                continue
            ref = reference.get((leagues[k], roles[k]))
            if ref is not None and len(ref) >= MIN_REFERENCE_N:
                lo, hi = bisect_left(ref, values[k]), bisect_right(ref, values[k])
                mid_rank = (lo + hi) / 2.0
                result[k] = min(100.0, max(0.0, 100.0 * (mid_rank - 0.5) / (len(ref) - 1)))
        for k in range(i, j):                                   # 2) only now extend the reference
            if np.isnan(values[k]) or _is_missing(roles[k]):
                continue
            insort(reference.setdefault((leagues[k], roles[k]), []), values[k])
        i = j
    return pd.Series(result, index=ordered.index).reindex(starters.index)


def aggregate_side_features(starters: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (game_id, side) with starter counts, strength evidence, XI / role / top-3 strengths.
    Needs: game_id, side, player_id, role, strength, value_age_days
    """
    key = ["game_id", "side"]
    starters = starters.copy()
    # the value age is averaged only over starters that actually received a strength value
    starters["value_age_days"] = starters["value_age_days"].where(starters["strength"].notna())
    grouped = starters.groupby(key)
    feats = pd.DataFrame({
        "starter_count": grouped["player_id"].size(),
        "strength_known_n": grouped["strength"].count(),
        "strength_age_mean_days": grouped["value_age_days"].mean(),
        "xi_strength": grouped["strength"].mean(),
    })
    feats["xi_strength"] = feats["xi_strength"].where(feats["strength_known_n"] >= MIN_KNOWN_FOR_XI)
    feats["strength_coverage"] = feats["strength_known_n"] / feats["starter_count"]

    for role, name in [("ATT", "attack_strength"), ("MID", "midfield_strength"),
                       ("DEF", "defence_strength"), ("GK", "gk_strength")]:
        feats[name] = starters[starters["role"] == role].groupby(key)["strength"].mean()

    known = starters[starters["strength"].notna()].sort_values("strength", ascending=False)
    top = known.groupby(key).head(3).groupby(key)["strength"].agg(["mean", "count"])
    feats["top3_strength"] = top["mean"].where(top["count"] >= MIN_KNOWN_FOR_TOP3)
    feats["top3_gap"] = feats["top3_strength"] - feats["xi_strength"]
    return feats.reset_index()


def side_features_to_wide(side_feats: pd.DataFrame, game_ids: pd.Series) -> pd.DataFrame:
    """Pivot (game, side) rows into home_/away_ columns; matches with no starters get zero counts."""
    metric_cols = [c for c in side_feats.columns if c not in ("game_id", "side")]
    out = pd.DataFrame({"game_id": game_ids.to_numpy()})
    for side in ("home", "away"):
        part = side_feats.loc[side_feats["side"] == side, ["game_id"] + metric_cols].copy()
        part.columns = ["game_id"] + [f"{side}_{c}" for c in metric_cols]
        out = out.merge(part, on="game_id", how="left")
        for c in (f"{side}_starter_count", f"{side}_strength_known_n"):
            out[c] = out[c].fillna(0).astype(int)
    for c in out.columns:
        if c != "game_id" and not c.endswith(("_starter_count", "_strength_known_n")):
            out[c] = out[c].round(ROUND_DECIMALS)
    return out


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """[VERIFIED against the engineered files] Home-minus-away differences and attack-vs-defence matchups."""
    df = df.copy()
    df["xi_strength_diff"] = df["home_xi_strength"] - df["away_xi_strength"]
    df["home_attack_vs_away_defence"] = df["home_attack_strength"] - df["away_defence_strength"]
    df["away_attack_vs_home_defence"] = df["away_attack_strength"] - df["home_defence_strength"]
    df["midfield_strength_diff"] = df["home_midfield_strength"] - df["away_midfield_strength"]
    df["gk_strength_diff"] = df["home_gk_strength"] - df["away_gk_strength"]
    for c in ["xi_strength_diff", "home_attack_vs_away_defence", "away_attack_vs_home_defence",
              "midfield_strength_diff", "gk_strength_diff"]:
        df[c] = df[c].round(ROUND_DECIMALS)
    return df


def add_player_feature_ready(df: pd.DataFrame) -> pd.DataFrame:
    """[VERIFIED range] Both XIs have 11 identified starters and both have coverage >= MIN_COVERAGE_READY."""
    df = df.copy()
    ready = ((df["home_starter_count"] == 11) & (df["away_starter_count"] == 11) &
             (df["home_strength_coverage"] >= MIN_COVERAGE_READY) &
             (df["away_strength_coverage"] >= MIN_COVERAGE_READY))
    df["player_feature_ready"] = ready.astype(int)
    return df


def finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Order columns like the common schema and format the date."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    missing = [c for c in COMMON_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns when finalising: {missing}")
    df = df[COMMON_COLUMNS].sort_values(["date", "league_code", "match_uid"]).reset_index(drop=True)
    return df


# ------------------------------------------------------------
# STAGE 1: EUROPEAN SOCCER SQLITE EXPORT
# ------------------------------------------------------------
def role_from_y(y: pd.Series) -> pd.Series:
    """[ASSUMPTION] goalkeeper / defence / midfield / attack from the vertical pitch coordinate."""
    y = pd.to_numeric(y, errors="coerce")
    role = pd.Series(np.select([y <= Y_GK_MAX, y <= Y_DEF_MAX, y <= Y_MID_MAX, y > Y_MID_MAX],
                               ["GK", "DEF", "MID", "ATT"], default=""), index=y.index, dtype=object)
    role[y.isna()] = None
    role[role == ""] = None
    return role


def formation_from_coordinates(match: pd.DataFrame, game_ids: pd.Series) -> pd.DataFrame:
    """
    [VERIFIED: 100% agreement with the existing Stage 1 formation labels]
    Label 'DEF-MID-ATT' (e.g. '4-4-2') counted from the 11 vertical pitch coordinates of a team.
    Issued only when all 11 Y values are present and exactly one is the goalkeeper (Y = 1).
    Uses the coordinates only (a missing player ID does not matter).
    `match` and `game_ids` must be row-aligned.
    """
    out = pd.DataFrame({"game_id": game_ids.to_numpy()})
    for side in ("home", "away"):
        y = match[[f"{side}_player_Y{i}" for i in range(1, 12)]].to_numpy(dtype=float)
        valid = ~np.isnan(y)
        n_gk = ((y >= 1) & (y <= Y_GK_MAX) & valid).sum(axis=1)
        n_def = ((y > Y_GK_MAX) & (y <= Y_DEF_MAX) & valid).sum(axis=1)
        n_mid = ((y > Y_DEF_MAX) & (y <= Y_MID_MAX) & valid).sum(axis=1)
        n_att = ((y > Y_MID_MAX) & valid).sum(axis=1)
        ok = (valid.sum(axis=1) == FULL_XI) & (n_gk == 1)
        label = pd.Series(n_def).astype(str) + "-" + pd.Series(n_mid).astype(str) + "-" + pd.Series(n_att).astype(str)
        out[f"{side}_formation"] = label.where(pd.Series(ok)).to_numpy()
    return out


def build_stage1(stage1_dir: Path) -> pd.DataFrame:
    print("\n=== STAGE 1 ===")
    stage1_dir = Path(stage1_dir)
    home_ids = [f"home_player_{i}" for i in range(1, 12)]
    away_ids = [f"away_player_{i}" for i in range(1, 12)]
    home_y = [f"home_player_Y{i}" for i in range(1, 12)]
    away_y = [f"away_player_Y{i}" for i in range(1, 12)]

    match = read_columns(find_csv(stage1_dir, "Match"),
                         ["match_api_id", "league_id", "season", "date", "home_team_api_id", "away_team_api_id",
                          "home_team_goal", "away_team_goal"] + home_ids + away_ids + home_y + away_y)
    league = read_columns(find_csv(stage1_dir, "League"), ["id", "country_id", "name"])
    country = read_columns(find_csv(stage1_dir, "country"), ["id", "name"])
    team = read_columns(find_csv(stage1_dir, "Team"), ["team_api_id", "team_long_name"])
    ratings = read_columns(find_csv(stage1_dir, "Player Attributes"), ["player_api_id", "date", "overall_rating"])
    print(f"Match {match.shape}, League {league.shape}, Country {country.shape}, Team {team.shape}, "
          f"Player Attributes {ratings.shape}")

    # ---- 1. match-level context -------------------------------------------------------------
    league = league.rename(columns={"id": "league_id", "name": "league_name"})
    country = country.rename(columns={"id": "country_id", "name": "country"})
    league = league.merge(country, on="country_id", how="left")
    league["league_code"] = league["league_name"].map(STAGE1_LEAGUE_CODES)
    unknown = league.loc[league["league_code"].isna(), "league_name"].tolist()
    if unknown:
        print("WARNING: leagues without a code mapping (they will be dropped):", unknown)

    m = match.merge(league[["league_id", "league_name", "country", "league_code"]], on="league_id", how="left")
    m = m[m["league_code"].notna()].copy()
    if KEEP_ONLY_PRIMARY_LEAGUES:
        m = m[m["league_code"].isin(PRIMARY_LEAGUES)].copy()
    names = team.set_index("team_api_id")["team_long_name"]
    m["home_team"] = m["home_team_api_id"].map(names)
    m["away_team"] = m["away_team_api_id"].map(names)
    m["date"] = to_ns(m["date"]).dt.normalize()
    m = m[m["home_team_goal"].notna() & m["away_team_goal"].notna()].copy()

    matches = pd.DataFrame({
        "game_id": m["match_api_id"].astype("int64"),
        "match_uid": "S1_" + m["match_api_id"].astype("int64").astype(str),
        "data_source": STAGE1_SOURCE,
        "strength_method": STAGE1_METHOD,
        "date": m["date"],
        "season": m["season"],
        "season_start": m["season"].str[:4].astype(int),
        "league_code": m["league_code"],
        "league_name": m["league_name"],
        "country": m["country"],
        "league_coverage_flag": "full_2008_2015",
        "home_team_id": m["home_team_api_id"].astype("int64"),
        "away_team_id": m["away_team_api_id"].astype("int64"),
        "home_team": m["home_team"],
        "away_team": m["away_team"],
        "home_goals": m["home_team_goal"].astype(int),
        "away_goals": m["away_team_goal"].astype(int),
    }).reset_index(drop=True)
    m = m.reset_index(drop=True)
    print("Matches kept:", len(matches))

    # ---- 2. starters in long format (one row per player slot) ------------------------------------
    parts = []
    for side, id_cols, y_cols in (("home", home_ids, home_y), ("away", away_ids, away_y)):
        for pid_col, y_col in zip(id_cols, y_cols):
            part = pd.DataFrame({"game_id": matches["game_id"].to_numpy(),
                                 "date": matches["date"].to_numpy(),
                                 "league_code": matches["league_code"].to_numpy(),
                                 "side": side,
                                 "player_id": m[pid_col].to_numpy(),
                                 "y": m[y_col].to_numpy()})
            parts.append(part)
    starters = pd.concat(parts, ignore_index=True)
    starters = starters[starters["player_id"].notna()].copy()
    starters["player_id"] = starters["player_id"].astype("int64")
    starters = starters.drop_duplicates(["game_id", "side", "player_id"])
    starters["role"] = role_from_y(starters["y"])

    # ---- 3. latest rating strictly before the match, then causal percentile ----------------------
    starters = _stage1_attach(starters, ratings)
    starters["raw_value"] = starters["overall_rating"]
    starters["strength"] = causal_percentile(starters, "raw_value")
    print(f"Starters: {len(starters):,}; with a rating: {starters['raw_value'].notna().mean():.1%}; "
          f"with a percentile: {starters['strength'].notna().mean():.1%}")

    # ---- 4. aggregate to match level -------------------------------------------------------------
    side_wide = side_features_to_wide(aggregate_side_features(starters), matches["game_id"])
    formation = formation_from_coordinates(m, matches["game_id"])
    form = add_form_features(matches)
    continuity = add_lineup_continuity(matches, starters)

    out = matches.copy()
    for block in (form, side_wide, continuity, formation):
        out = out.merge(block, on="game_id", how="left")
    out = add_matchup_features(out)
    out = add_player_feature_ready(out)
    out["audit_home_goals"] = out["home_goals"]
    out["audit_away_goals"] = out["away_goals"]
    out["target_result"] = result_from_goals(out["home_goals"], out["away_goals"])
    return finalise(out)


def _stage1_attach(starters: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest pre-match FIFA overall_rating to each Stage 1 starter."""
    r = ratings.rename(columns={"player_api_id": "player_id", "date": "rating_date"})
    r = r[r["overall_rating"].notna()]
    return attach_latest_value(starters, r, id_col="player_id", value_date_col="rating_date",
                               value_col="overall_rating")


# ------------------------------------------------------------
# STAGE 2: TRANSFERMARKT BUNDLE
# ------------------------------------------------------------
def role_from_position(position: pd.Series) -> pd.Series:
    """Transfermarkt position text -> GK / DEF / MID / ATT (order matters: 'Defensive Midfield' is a midfielder)."""
    text = position.astype("string").str.lower()
    role = pd.Series(pd.NA, index=position.index, dtype="object")
    known = text.notna()
    role[known] = "ATT"                                            # winger, striker, forward, attack ...
    role[known & text.str.contains("back|defender|sweeper", na=False)] = "DEF"
    role[known & text.str.contains("mid", na=False)] = "MID"
    role[known & text.str.contains("keeper", na=False)] = "GK"
    return role


def build_stage2(stage2_dir: Path, first_season: int = 2016) -> pd.DataFrame:
    print("\n=== STAGE 2 ===")
    stage2_dir = Path(stage2_dir)
    games = read_columns(find_csv(stage2_dir, "games"),
                         ["game_id", "competition_id", "season", "date", "home_club_id", "away_club_id",
                          "home_club_goals", "away_club_goals", "home_club_formation", "away_club_formation",
                          "home_club_name", "away_club_name"])
    comps = read_columns(find_csv(stage2_dir, "competitions"), ["competition_id", "name", "country_name"])
    values = read_columns(find_csv(stage2_dir, "player_valuations"), ["player_id", "date", "market_value_in_eur"])

    # ---- 1. match-level context -------------------------------------------------------------
    # Games from REFERENCE_FIRST_SEASON_STAGE2 onward are loaded so that the percentile reference has a
    # warm-up period (2012-2015 lineups); only games from `first_season` onward become output rows.
    league_codes = PRIMARY_LEAGUES if KEEP_ONLY_PRIMARY_LEAGUES else STAGE2_LEAGUE_CODES
    g_all = games[games["competition_id"].isin(league_codes) &
                  (games["season"] >= min(first_season, REFERENCE_FIRST_SEASON_STAGE2))].copy()
    g_all = g_all[g_all["home_club_goals"].notna() & g_all["away_club_goals"].notna()]
    g_all = g_all.merge(comps.rename(columns={"name": "league_name", "country_name": "country"}),
                        on="competition_id", how="left")
    g_all["date"] = to_ns(g_all["date"]).dt.normalize()
    g_all = g_all.reset_index(drop=True)
    g = g_all[g_all["season"] >= first_season].reset_index(drop=True)
    first_seen = g.groupby("competition_id")["season"].transform("min")
    coverage_flag = np.where(first_seen >= 2024, "partial_2024_2025", "full_2016_2025")

    matches = pd.DataFrame({
        "game_id": g["game_id"].astype("int64"),
        "match_uid": "TM_" + g["game_id"].astype("int64").astype(str),
        "data_source": STAGE2_SOURCE,
        "strength_method": STAGE2_METHOD,
        "date": g["date"],
        "season": g["season"].astype(int).map(season_label_short),
        "season_start": g["season"].astype(int),
        "league_code": g["competition_id"],
        "league_name": g["league_name"],
        "country": g["country"],
        "league_coverage_flag": coverage_flag,
        "home_team_id": g["home_club_id"].astype("int64"),
        "away_team_id": g["away_club_id"].astype("int64"),
        "home_team": g["home_club_name"],
        "away_team": g["away_club_name"],
        "home_goals": g["home_club_goals"].astype(int),
        "away_goals": g["away_club_goals"].astype(int),
        "home_formation": g["home_club_formation"],
        "away_formation": g["away_club_formation"],
    })
    print("Output matches:", len(matches), "| matches used for the percentile reference:", len(g_all))
    print(matches["league_code"].value_counts().to_string())

    # ---- 2. starting lineups (large file: read in chunks and keep only what is needed) -----------
    path = find_csv(stage2_dir, "game_lineups")
    wanted_games = set(g_all["game_id"])
    kept = []
    for chunk in pd.read_csv(path, usecols=["game_id", "club_id", "type", "player_id", "position"],
                             chunksize=500_000):
        chunk = chunk[(chunk["type"] == "starting_lineup") & chunk["game_id"].isin(wanted_games)]
        kept.append(chunk)
    lineups = pd.concat(kept, ignore_index=True)
    sides = pd.DataFrame({"game_id": g_all["game_id"].astype("int64"), "date": g_all["date"],
                          "league_code": g_all["competition_id"],
                          "home_team_id": g_all["home_club_id"].astype("int64"),
                          "away_team_id": g_all["away_club_id"].astype("int64")})
    starters = lineups.merge(sides, on="game_id", how="inner")
    starters["side"] = np.where(starters["club_id"] == starters["home_team_id"], "home",
                        np.where(starters["club_id"] == starters["away_team_id"], "away", ""))
    starters = starters[starters["side"] != ""].dropna(subset=["player_id"]).copy()
    starters["player_id"] = starters["player_id"].astype("int64")
    starters = starters.drop_duplicates(["game_id", "side", "player_id"])
    starters["role"] = role_from_position(starters["position"])
    starters = starters[["game_id", "date", "league_code", "side", "player_id", "role"]]

    # ---- 3. latest valuation strictly before the match -> log1p -> causal percentile -----------------
    v = values[values["market_value_in_eur"].notna()].rename(columns={"date": "valuation_date"})
    starters = attach_latest_value(starters, v, id_col="player_id", value_date_col="valuation_date",
                                   value_col="market_value_in_eur")
    starters["raw_value"] = np.log1p(starters["market_value_in_eur"].astype(float))
    starters["strength"] = causal_percentile(starters, "raw_value")
    starters = starters[starters["game_id"].isin(set(matches["game_id"]))].copy()   # keep output matches only
    print(f"Starters (output matches): {len(starters):,}; with a valuation: {starters['raw_value'].notna().mean():.1%}; "
          f"with a percentile: {starters['strength'].notna().mean():.1%}")

    # ---- 4. aggregate to match level -------------------------------------------------------------
    side_wide = side_features_to_wide(aggregate_side_features(starters), matches["game_id"])
    form = add_form_features(matches)
    continuity = add_lineup_continuity(matches, starters)

    out = matches.copy()
    for block in (form, side_wide, continuity):
        out = out.merge(block, on="game_id", how="left")
    out = add_matchup_features(out)
    out = add_player_feature_ready(out)
    out["audit_home_goals"] = out["home_goals"]
    out["audit_away_goals"] = out["away_goals"]
    out["target_result"] = result_from_goals(out["home_goals"], out["away_goals"])
    return finalise(out)


# ------------------------------------------------------------
# MASTER AND PRIMARY DATASETS  (same logic as your master-creation script)
# ------------------------------------------------------------
def build_master_and_primary(stage1: pd.DataFrame, stage2: pd.DataFrame, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in (("Stage 1", stage1), ("Stage 2", stage2)):
        missing = [c for c in COMMON_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
    stage1 = stage1[COMMON_COLUMNS].copy()
    stage2 = stage2[COMMON_COLUMNS].copy()
    for df in (stage1, stage2):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["season_start"] = pd.to_numeric(df["season_start"], errors="coerce")
        df["target_result"] = df["target_result"].astype("string").str.upper().str.strip()

    master = pd.concat([stage1, stage2], axis=0, ignore_index=True)
    master = master.sort_values(["date", "league_code", "match_uid"]).reset_index(drop=True)

    print("\nMASTER DATASET:", master.shape)
    print(master["data_source"].value_counts(dropna=False).to_string())
    print(master["target_result"].value_counts(dropna=False).to_string())
    print("Date range:", master["date"].min(), "to", master["date"].max())

    unexpected = set(master["target_result"].dropna().unique()) - {"H", "D", "A"}
    if unexpected:
        raise ValueError(f"Unexpected target values: {unexpected}")
    if master["match_uid"].duplicated().any():
        master[master["match_uid"].duplicated(keep=False)].to_csv(output_dir / "duplicate_match_ids.csv", index=False)
        raise ValueError("Duplicate match_uid values detected; see duplicate_match_ids.csv")

    if SAVE_MASTER_FILE:
        master.to_csv(output_dir / MASTER_FILE, index=False)
        print("Saved:", output_dir / MASTER_FILE)

    primary = master[master["league_code"].isin(PRIMARY_LEAGUES)].copy()
    primary = primary.sort_values(["date", "league_code", "match_uid"]).reset_index(drop=True)
    print("\nPRIMARY NINE-LEAGUE DATASET:", primary.shape)
    print(primary["league_code"].value_counts().to_string())
    print(primary["target_result"].value_counts(normalize=True).round(4).to_string())
    primary.to_csv(output_dir / PRIMARY_FILE, index=False)
    print("Saved:", output_dir / PRIMARY_FILE)
    return master, primary


# ------------------------------------------------------------
# VALIDATION AGAINST YOUR EXISTING ENGINEERED FILES
# ------------------------------------------------------------
def compare_with_existing(rebuilt: pd.DataFrame, existing_path: Path, label: str) -> pd.DataFrame:
    """Report how closely a rebuilt stage file agrees with an engineered file you already have."""
    old = pd.read_csv(existing_path)
    new = rebuilt.copy()
    print(f"\n--- VALIDATION: {label} ---")
    print(f"rows rebuilt = {len(new):,} | rows existing = {len(old):,}")
    ids_new, ids_old = set(new["game_id"]), set(old["game_id"])
    print(f"game_id only in rebuilt: {len(ids_new - ids_old):,} | only in existing: {len(ids_old - ids_new):,}")
    both = new.merge(old, on="game_id", suffixes=("_new", "_old"))
    print(f"matches compared: {len(both):,}")

    rows = []

    def exact(col):
        a, b = both[f"{col}_new"], both[f"{col}_old"]
        same = (a.isna() & b.isna()) | (a.astype(str) == b.astype(str))
        rows.append((col, "exact", float(same.mean())))

    def close(col, tol=1e-3):
        a, b = both[f"{col}_new"].astype(float), both[f"{col}_old"].astype(float)
        ok = (a.isna() & b.isna()) | (np.abs(a - b) <= tol)
        rows.append((col, f"within {tol}", float(ok.mean())))

    for c in ["date", "league_code", "home_team_id", "away_team_id", "audit_home_goals", "audit_away_goals",
              "target_result", "home_starter_count", "away_starter_count", "home_strength_known_n",
              "away_strength_known_n", "home_form_n_L5", "away_form_n_L5", "home_formation", "away_formation",
              "player_feature_ready"]:
        exact(c)
    for c in ["home_points_per_game_L5", "away_win_rate_L5", "home_goal_diff_avg_L5", "home_lineup_continuity",
              "away_lineup_continuity", "home_strength_coverage", "home_strength_age_mean_days"]:
        close(c, 1e-3 if "age" not in c else 1.0)
    report = pd.DataFrame(rows, columns=["column", "comparison", "share_agreeing"])

    # strength columns are percentile-based: report agreement as correlation and mean absolute difference
    corr_rows = []
    for c in ["home_xi_strength", "away_xi_strength", "home_attack_strength", "home_gk_strength", "xi_strength_diff"]:
        a, b = both[f"{c}_new"].astype(float), both[f"{c}_old"].astype(float)
        mask = a.notna() & b.notna()
        both_missing = float((a.isna() & b.isna()).mean())
        corr_rows.append((c, float(a[mask].corr(b[mask], method="spearman")) if mask.sum() > 2 else np.nan,
                          float((a[mask] - b[mask]).abs().mean()) if mask.any() else np.nan,
                          float(mask.mean()), both_missing))
    corr = pd.DataFrame(corr_rows, columns=["column", "spearman_corr", "mean_abs_diff", "share_both_present",
                                            "share_both_missing"])
    print(report.round(4).to_string(index=False))
    print(corr.round(4).to_string(index=False))
    print("Guide: the form, goals, target and starter-count columns should agree ~100%. The strength columns "
          "depend on the percentile design (MIN_REFERENCE_N, role bands); a Spearman correlation close to 1 "
          "means the same ordering of teams even if the 0-100 values differ slightly.")
    return report


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the football master dataset from raw CSV folders.")
    parser.add_argument("--stage1-dir", default="csv/stage_1")
    parser.add_argument("--stage2-dir", default="csv/stage_2")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--existing-stage1", default=None, help="existing engineered Stage 1 CSV (validation)")
    parser.add_argument("--existing-stage2", default=None, help="existing engineered Stage 2 CSV (validation)")
    parser.add_argument("--all-leagues", action="store_true",
                        help="keep all 11 leagues in the stage files and master (default: the 9 primary leagues)")
    parser.add_argument("--save-master", action="store_true", help="also save the Master CSV (default: primary only)")
    args = parser.parse_args(argv)
    global KEEP_ONLY_PRIMARY_LEAGUES, SAVE_MASTER_FILE
    SAVE_MASTER_FILE = args.save_master
    KEEP_ONLY_PRIMARY_LEAGUES = not args.all_leagues

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1 = build_stage1(Path(args.stage1_dir))
    stage1.to_csv(out_dir / STAGE1_FILE, index=False)
    print("Saved:", out_dir / STAGE1_FILE, stage1.shape)

    stage2 = build_stage2(Path(args.stage2_dir))
    stage2.to_csv(out_dir / STAGE2_FILE, index=False)
    print("Saved:", out_dir / STAGE2_FILE, stage2.shape)

    build_master_and_primary(stage1, stage2, out_dir)

    if args.existing_stage1:
        compare_with_existing(stage1, Path(args.existing_stage1), "Stage 1").to_csv(
            out_dir / "validation_stage1.csv", index=False)
    if args.existing_stage2:
        compare_with_existing(stage2, Path(args.existing_stage2), "Stage 2").to_csv(
            out_dir / "validation_stage2.csv", index=False)
    print("\nDATA ENGINEERING COMPLETED.")


if __name__ == "__main__":
    sys.exit(main())
