#!/usr/bin/env python3
# ============================================================
# SELF-TEST FOR Build_Master_Dataset.py
#
# PART A (real data): re-derives columns of your ENGINEERED files with the pipeline's helper functions.
#   Run it with:  python selftest_build.py --stage2 <engineered Stage 2 csv> --stage1 <engineered Stage 1 csv>
# PART B (synthetic raw data): builds tiny fake raw CSVs in the exact raw formats, runs the whole
#   pipeline and checks schema, causality and the output files. This proves the code RUNS, not that
#   it reproduces your numbers; use --existing-stage1/2 in the main script for that.
# ============================================================
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import Build_Master_Dataset as B


# ----------------------------- PART A -----------------------------
def part_a(path, label):
    d = pd.read_csv(path, parse_dates=["date"])
    m = d.rename(columns={"audit_home_goals": "home_goals", "audit_away_goals": "away_goals"})
    form = B.add_form_features(m[["game_id", "date", "season_start", "home_team_id", "away_team_id",
                                  "home_goals", "away_goals"]])
    both = d.merge(form, on="game_id", suffixes=("_old", "_new"))
    cols = [c for c in form.columns if c != "game_id"]
    worst = 1.0
    for c in cols:
        a, b = both[f"{c}_old"].astype(float), both[f"{c}_new"].astype(float)
        ok = ((a.isna() & b.isna()) | (np.abs(a - b) <= 1e-4)).mean()
        worst = min(worst, ok)
    print(f"[A] {label}: form features agree with the engineered file on {worst:.4%} of rows (worst column)")
    assert worst > 0.9999

    mu = B.add_matchup_features(d.drop(columns=["xi_strength_diff", "home_attack_vs_away_defence",
                                                "away_attack_vs_home_defence", "midfield_strength_diff",
                                                "gk_strength_diff"]))
    for c in ["xi_strength_diff", "home_attack_vs_away_defence", "away_attack_vs_home_defence",
              "midfield_strength_diff", "gk_strength_diff"]:
        a, b = d[c].astype(float), mu[c].astype(float)
        ok = ((a.isna() & b.isna()) | (np.abs(a - b) <= 2e-4)).mean()
        assert ok > 0.9999, (c, ok)
    print(f"[A] {label}: matchup features agree with the engineered file (tolerance 2e-4)")

    ready = B.add_player_feature_ready(d.drop(columns=["player_feature_ready"]))
    agree = (ready["player_feature_ready"] == d["player_feature_ready"]).mean()
    print(f"[A] {label}: player_feature_ready rule agrees on {agree:.4%} of rows")
    assert agree > 0.9999

    gap = (d["home_top3_strength"] - d["home_xi_strength"]).round(4)
    ok = ((gap.isna() & d["home_top3_gap"].isna()) | (np.abs(gap - d["home_top3_gap"]) <= 2e-4)).mean()
    assert ok > 0.9999
    print(f"[A] {label}: top3_gap = top3_strength - xi_strength agrees on {ok:.4%} of rows")


# ----------------------------- PART B -----------------------------
def make_stage1_raw(folder: Path, seed=1, n_days=60):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1, 2], "name": ["England", "Poland"]}).to_csv(folder / "country.csv", index=False)
    pd.DataFrame({"id": [10, 20], "country_id": [1, 2],
                  "name": ["England Premier League", "Poland Ekstraklasa"]}).to_csv(folder / "League.csv", index=False)
    teams = {10: [100 + i for i in range(6)], 20: [200 + i for i in range(6)]}
    pd.DataFrame({"team_api_id": sum(teams.values(), []),
                  "team_long_name": [f"Team {t}" for t in sum(teams.values(), [])]}).to_csv(folder / "Team.csv", index=False)
    players = list(range(1000, 1300))
    # rating history: every player rated at 3 dates before and during the period
    rat = []
    for p in players:
        for d in ["2008-01-01", "2008-08-01", "2008-10-01"]:
            rat.append((p, d + " 00:00:00", int(rng.integers(55, 90))))
    pd.DataFrame(rat, columns=["player_api_id", "date", "overall_rating"]).to_csv(folder / "Player Attributes.csv", index=False)

    ys = [1, 3, 3, 3, 3, 7, 7, 7, 7, 11, 11]            # a 4-4-2
    rows, mid = [], 5000
    start = pd.Timestamp("2008-08-09")
    for day in range(n_days):
        date = start + pd.Timedelta(days=7 * day)
        for league, ts in teams.items():
            order = rng.permutation(ts)
            for k in range(0, 6, 2):
                h, a = int(order[k]), int(order[k + 1])
                row = {"match_api_id": mid, "league_id": league, "season": "2008/2009" if day < 30 else "2009/2010",
                       "date": date.strftime("%Y-%m-%d") + " 00:00:00", "home_team_api_id": h, "away_team_api_id": a,
                       "home_team_goal": int(rng.poisson(1.5)), "away_team_goal": int(rng.poisson(1.1))}
                for side in ("home", "away"):
                    ids = rng.choice(players, 11, replace=False)
                    for i in range(11):
                        row[f"{side}_player_{i + 1}"] = int(ids[i]) if rng.random() > 0.03 else np.nan
                        row[f"{side}_player_Y{i + 1}"] = ys[i]
                rows.append(row); mid += 1
    pd.DataFrame(rows).to_csv(folder / "Match.csv", index=False)
    pd.DataFrame({"name": ["Team", "Match"], "seq": [1, 2]}).to_csv(folder / "sqlite sequnce.csv", index=False)


def make_stage2_raw(folder: Path, seed=2, n_days=60):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"competition_id": ["GB1", "PL1"], "name": ["premier-league", "pko-bp-ekstraklasa"],
                  "country_name": ["England", "Poland"]}).to_csv(folder / "competitions.csv", index=False)
    teams = {"GB1": [10 + i for i in range(6)], "PL1": [50 + i for i in range(6)]}
    players = list(range(5000, 5300))
    pd.DataFrame({"player_id": np.repeat(players, 2),
                  "date": ["2016-01-15", "2016-09-20"] * len(players),
                  "market_value_in_eur": rng.integers(100_000, 60_000_000, 2 * len(players))}).to_csv(
        folder / "player_valuations.csv", index=False)
    positions = ["Goalkeeper", "Left-Back", "Centre-Back", "Centre-Back", "Right-Back", "Defensive Midfield",
                 "Central Midfield", "Left Winger", "Right Winger", "Attacking Midfield", "Centre-Forward"]
    games, lineups, gid = [], [], 9000
    start = pd.Timestamp("2016-08-13")
    for day in range(n_days):
        date = start + pd.Timedelta(days=7 * day)
        for comp, ts in teams.items():
            order = rng.permutation(ts)
            for k in range(0, 6, 2):
                h, a = int(order[k]), int(order[k + 1])
                season = 2016 if day < 30 else 2017
                games.append({"game_id": gid, "competition_id": comp, "season": season, "date": date.strftime("%Y-%m-%d"),
                              "home_club_id": h, "away_club_id": a, "home_club_goals": int(rng.poisson(1.5)),
                              "away_club_goals": int(rng.poisson(1.1)), "home_club_formation": "4-3-3 Attacking",
                              "away_club_formation": "4-4-2", "home_club_name": f"Club {h}", "away_club_name": f"Club {a}"})
                for club in (h, a):
                    ids = rng.choice(players, 11, replace=False)
                    for i in range(11):
                        lineups.append({"game_id": gid, "club_id": club, "type": "starting_lineup",
                                        "player_id": int(ids[i]), "position": positions[i]})
                    lineups.append({"game_id": gid, "club_id": club, "type": "substitutes",
                                    "player_id": int(rng.choice(players)), "position": "Centre-Forward"})
                gid += 1
    pd.DataFrame(games).to_csv(folder / "games.csv", index=False)
    pd.DataFrame(lineups).to_csv(folder / "game_lineups.csv", index=False)
    for extra in ["appearances", "club_games", "clubs", "countries", "game_events", "national_teams", "players", "transfers"]:
        pd.DataFrame({"x": [1]}).to_csv(folder / f"{extra}.csv", index=False)       # unused files must not matter


def part_b():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_stage1_raw(tmp / "stage_1")
        make_stage2_raw(tmp / "stage_2")
        old_min = B.MIN_REFERENCE_N
        B.KEEP_ONLY_PRIMARY_LEAGUES = False          # the synthetic data includes PL1 to test the primary filter
        B.MIN_REFERENCE_N = 30
        s1 = B.build_stage1(tmp / "stage_1")
        s2 = B.build_stage2(tmp / "stage_2")
        master, primary = B.build_master_and_primary(s1, s2, tmp / "out")

        # --- schema and basic integrity
        for name, df in (("stage1", s1), ("stage2", s2), ("master", master), ("primary", primary)):
            assert list(df.columns) == B.COMMON_COLUMNS, name
            assert not df["match_uid"].duplicated().any(), name
            assert set(df["target_result"].dropna()) <= {"H", "D", "A"}, name
        assert set(primary["league_code"]) <= set(B.PRIMARY_LEAGUES)
        assert "PL1" not in set(primary["league_code"])
        assert len(master) == len(s1) + len(s2)
        assert (tmp / "out" / B.PRIMARY_FILE).exists()
        # --- target consistent with goals
        derived = np.where(s2.audit_home_goals > s2.audit_away_goals, "H",
                  np.where(s2.audit_home_goals == s2.audit_away_goals, "D", "A"))
        assert (derived == s2.target_result).all()
        # --- derived relations
        ok = s2.dropna(subset=["home_xi_strength", "away_xi_strength"])
        assert np.allclose(ok.home_xi_strength - ok.away_xi_strength, ok.xi_strength_diff, atol=2e-4)
        assert (s2.home_strength_coverage.dropna().between(0, 1)).all()
        assert (s2.home_xi_strength.dropna().between(0, 100)).all()
        assert s1["home_formation"].dropna().eq("4-4-2").all()          # all synthetic XIs are 4-4-2
        assert s1["league_code"].isin(["GB1", "PL1"]).all()
        print("[B] schema, ids, target, ranges, formation label: OK")
        print("[B] stage1 rows:", len(s1), "| stage2 rows:", len(s2), "| xi_strength present:",
              f"{s1.home_xi_strength.notna().mean():.0%} / {s2.home_xi_strength.notna().mean():.0%}")

        # --- first-season matches have no history, later matches do
        # a team's first league match of a season (home OR away) must have no prior form
        h = s2[["game_id", "date", "season_start", "home_team_id", "home_form_n_L5"]].set_axis(
            ["game_id", "date", "season_start", "team", "n"], axis=1)
        a = s2[["game_id", "date", "season_start", "away_team_id", "away_form_n_L5"]].set_axis(
            ["game_id", "date", "season_start", "team", "n"], axis=1)
        long = pd.concat([h, a]).sort_values(["team", "season_start", "date", "game_id"])
        first = long.groupby(["team", "season_start"]).head(1)
        assert (first["n"] == 0).all()
        assert (long.groupby(["team", "season_start"])["n"].max() == 5).all()   # window caps at 5

        # --- CAUSALITY TEST 1: changing a LATER match must not change EARLIER features
        s2_alt_dir = tmp / "stage_2_alt"
        make_stage2_raw(s2_alt_dir)
        games = pd.read_csv(s2_alt_dir / "games.csv")
        cutoff = sorted(games["date"].unique())[40]
        late = games["date"] >= cutoff
        games.loc[late, "home_club_goals"] += 5                              # alter late results
        lu = pd.read_csv(s2_alt_dir / "game_lineups.csv")
        late_ids = set(games.loc[late, "game_id"])
        lu.loc[lu["game_id"].isin(late_ids), "player_id"] = 5000            # alter late lineups
        games.to_csv(s2_alt_dir / "games.csv", index=False)
        lu.to_csv(s2_alt_dir / "game_lineups.csv", index=False)
        s2_alt = B.build_stage2(s2_alt_dir)
        early_a = s2[s2["date"] < cutoff].set_index("match_uid")
        early_b = s2_alt[s2_alt["date"] < cutoff].set_index("match_uid")
        feature_cols = [c for c in B.COMMON_COLUMNS if c not in ("match_uid", "audit_home_goals", "audit_away_goals", "target_result")]
        pd.testing.assert_frame_equal(early_a[feature_cols], early_b[feature_cols])
        print("[B] causality: altering later results / lineups does not change any earlier match's features: OK")

        # --- CAUSALITY TEST 2: strength uses only values dated strictly before the match
        starters_check = B.attach_latest_value(
            pd.DataFrame({"game_id": [1, 2], "date": ["2016-09-20", "2016-09-21"], "player_id": [7, 7],
                          "league_code": ["GB1", "GB1"], "role": ["MID", "MID"]}),
            pd.DataFrame({"player_id": [7, 7], "valuation_date": ["2016-01-01", "2016-09-20"],
                          "market_value_in_eur": [1.0, 2.0]}),
            id_col="player_id", value_date_col="valuation_date", value_col="market_value_in_eur")
        got = starters_check.sort_values("game_id")["market_value_in_eur"].tolist()
        assert got == [1.0, 2.0], got            # same-day valuation is NOT used; next-day match uses it
        print("[B] causality: a valuation dated on the match day is not used (strictly earlier only): OK")

        # --- percentile: same-date matches are scored before the reference set is extended
        n = 40
        df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"] * n + ["2020-01-02"] * 5),
                           "league_code": "GB1", "role": "MID", "raw_value": list(np.arange(n)) + [5, 20, 39, 100, -5]})
        pct = B.causal_percentile(df)
        assert pct.iloc[:n].isna().all()                                   # day 1: no reference yet
        assert pct.iloc[n + 2] > pct.iloc[n + 1] > pct.iloc[n]             # monotone in the raw value
        assert pct.iloc[n + 3] > 95 and pct.iloc[n + 4] < 5
        print("[B] percentile: no reference on day 1, monotone and bounded afterwards: OK")
        B.MIN_REFERENCE_N = old_min

        # --- validation routine runs against a (here: identical) existing file
        s2.to_csv(tmp / "existing_s2.csv", index=False)
        rep = B.compare_with_existing(s2, tmp / "existing_s2.csv", "synthetic self-comparison")
        assert (rep["share_agreeing"] > 0.999).all()
        print("[B] validation report: OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=None, help="engineered Stage 1 CSV (optional real-data check)")
    ap.add_argument("--stage2", default=None, help="engineered Stage 2 CSV (optional real-data check)")
    args = ap.parse_args()
    if args.stage1:
        part_a(args.stage1, "Stage 1")
    if args.stage2:
        part_a(args.stage2, "Stage 2")
    part_b()
    print("\nALL SELF-TESTS PASSED")
