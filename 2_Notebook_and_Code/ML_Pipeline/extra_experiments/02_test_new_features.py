"""Test whether the extra pre-match features (script 01) improve the frozen final model.

The frozen configuration (Random Forest, FS3 settings) is retrained with and without each feature group.
Includes a brute-force leakage check (200 random starters). Output: ml_outputs/new_feature_experiment.csv.

Run from this folder:  python 02_test_new_features.py --stage2-dir "<path to csv/stage_2>"
"""
import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.metrics import f1_score, log_loss
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ML_DIR = HERE.parent
ap = argparse.ArgumentParser()
ap.add_argument("--stage2-dir", default="csv/stage_2")
ap.add_argument("--features", default=str(HERE / "new_features_tm.csv"))
ap.add_argument("--out-dir", default=str(ML_DIR / "ml_outputs"))
args = ap.parse_args()
U, OUT = Path(args.stage2_dir), Path(args.out_dir)

frozen = joblib.load(ML_DIR / "ml_outputs" / "final_model.joblib")            # template only (clone = same settings, unfitted)
FS3 = pd.read_csv(ML_DIR / "ml_outputs" / "final_feature_list.csv").feature.tolist()
P = pd.read_csv(ML_DIR / "European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv", low_memory=False, parse_dates=["date"])
NF = pd.read_csv(args.features, parse_dates=["date"])
new = [c for c in NF.columns if c not in ("match_uid", "game_id", "date", "home_team_id", "away_team_id")]
D = P.merge(NF[["match_uid"] + new], on="match_uid", how="left")

# Leakage check: a starter's goals over the previous 5 appearances, rebuilt by brute force from strictly earlier dates
a = pd.read_csv(U / "appearances.csv", usecols=["game_id", "player_id", "date", "goals"], parse_dates=["date"]).sort_values(["player_id", "date", "game_id"])
st = []
for ch in pd.read_csv(U / "game_lineups.csv", usecols=["game_id", "player_id", "club_id", "type"], chunksize=500000):
    st.append(ch[(ch.type == "starting_lineup") & ch.game_id.isin(set(NF.game_id))])
st = pd.concat(st).merge(NF[["game_id", "date"]], on="game_id")
ok, byp = 0, dict(tuple(a.groupby("player_id")))
for _, r in st.sample(200, random_state=1).iterrows():
    h = byp.get(r.player_id)
    if h is None:
        ok += 1
        continue
    prev = h[h.date < r.date].tail(5).goals.sum()
    cur = h[h.game_id == r.game_id]
    idx = h.index.get_loc(cur.index[0]) if len(cur) else None
    built = h.iloc[max(0, idx - 5):idx].goals.sum() if idx is not None else prev
    ok += int(built == prev)
print(f"Leakage check (build vs brute force, strictly earlier dates): {ok}/200 identical")

GROUPS = {"rest": ["home_rest_days", "away_rest_days", "home_matches_prev14", "away_matches_prev14", "rest_days_diff"],
          "discipline": ["home_yellow_L5", "away_yellow_L5", "home_red_L5", "away_red_L5", "home_red_prev_match", "away_red_prev_match"],
          "player_form": ["home_xi_goals_L5", "away_xi_goals_L5", "home_xi_assists_L5", "away_xi_assists_L5",
                          "home_xi_minutes_L5", "away_xi_minutes_L5", "xi_goals_L5_diff"]}
ALL_NEW = sum(GROUPS.values(), [])


def model_for(feats):
    m = clone(frozen)
    prep = m.named_steps["prep"]
    prep.transformers = [("num", prep.transformers[0][1], [f for f in feats if f != "league_code"]),
                         ("cat", prep.transformers[1][1], ["league_code"])]
    return m


def run(name, feats, data):
    tr, va, te = data[data.season_start <= 2022], data[data.season_start == 2023], data[data.season_start >= 2024]
    m = model_for(feats).fit(tr[feats], tr.target_result)
    r = {"variant": name, "n_features": len(feats), "train_rows": len(tr)}
    for part, d in (("val", va), ("test", te)):
        p = m.predict_proba(d[feats]); pr = m.classes_[p.argmax(1)]
        r[f"{part}_macro_f1"] = f1_score(d.target_result, pr, average="macro")
        r[f"{part}_log_loss"] = log_loss(d.target_result, p, labels=m.classes_)
        r[f"{part}_draw_f1"] = f1_score(d.target_result, pr, labels=["D"], average="macro")
        if part == "test":
            r["_pred"], r["_y"] = pr, d.target_result.values
    return r


modern = D[D.season_start >= 2016]
runs = [run("A0 FS3 (modern 2016-2022 training)", FS3, modern),
        run("A1 FS3 + rest days", FS3 + GROUPS["rest"], modern),
        run("A2 FS3 + discipline", FS3 + GROUPS["discipline"], modern),
        run("A3 FS3 + player form", FS3 + GROUPS["player_form"], modern),
        run("A4 FS3 + all new features", FS3 + ALL_NEW, modern),
        run("B0 FS3 (full 2008-2022 training, official setup)", FS3, D),
        run("B1 FS3 + all new (full history; 2008-2016 values imputed)", FS3 + ALL_NEW, D)]

# Paired bootstrap on the same test matches (1,000 resamples, seed 42)
rng = np.random.default_rng(42); y = runs[0]["_y"]
idx = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
rows = []
for r in runs:
    base = runs[0] if r["variant"].startswith("A") else runs[5]
    d = np.array([f1_score(y[i], r["_pred"][i], average="macro") - f1_score(y[i], base["_pred"][i], average="macro") for i in idx])
    rows.append({**{k: v for k, v in r.items() if not k.startswith("_")},
                 "val_gain": r["val_macro_f1"] - base["val_macro_f1"], "test_gain": r["test_macro_f1"] - base["test_macro_f1"],
                 "test_gain_95CI": "—" if r is base else f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"})
res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(res[["variant", "n_features", "val_macro_f1", "val_gain", "test_macro_f1", "test_gain", "test_gain_95CI"]].round(4).to_string(index=False))
OUT.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT / "new_feature_experiment.csv", index=False)
print("Saved", OUT / "new_feature_experiment.csv")
