"""문제 정형화 덱의 야구 숫자 → assets/baseball.json. 실습 노트북 (lab/make_lab.py) 과 같은 데이터 · 같은 설정.
  투구 (Baseball Savant / Statcast, 2024-06 정규시즌 모든 투구, 5일씩 6번 받는다): 한 행 = 공 하나
     전부 '홈런 아님' vs 규칙 '3볼 (3-1 · 3-2) 이면 홈런', 볼카운트별 홈런 비율, 한 행 = 공 vs 타석 vs 타자, 오타니 홈런 타석 예시
  던지기 전 두 문제 (같은 투구, 한 행 = 공 하나, 입력 = 던지기 직전 볼카운트 · 타자 · 투수, 원-핫):
     문제 A = 타자가 이 공에 스윙할까 (분류, 정확도), 문제 B = 이 공의 구속 km/h (회귀, MAE)
     날짜 순서로 나눔: 6/1~23 학습, 6/24~30 시험. 로지스틱회귀 / 릿지 (alpha=1).
     기준선 = 늘 '안 친다' / 학습 평균 구속. 반칙 예: 구종 (던진 뒤에야 안다) 을 입력에 넣으면 B 의 MAE
실행: python3 exp_baseball.py  (처음엔 투구 CSV 6개 받는 데 4~5분)"""
import json, os, urllib.request
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "lab", "data"); os.makedirs(D, exist_ok=True)
UA = {"User-Agent": "book-ml-course"}
SAVANT = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details&player_type=batter"
          "&hfGT=R%7C&hfSea=2024%7C&game_date_gt={}&game_date_lt={}")      # 한 번에 25,000줄까지라 5일씩
CHUNKS = [("01", "05"), ("06", "10"), ("11", "15"), ("16", "20"), ("21", "25"), ("26", "30")]
OHTANI = 660271


def fetch(url, path):
    if not os.path.exists(path):
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


J = {}
mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

# ---- 0. 투구: 한 행 = 공 하나
pt = pd.concat([pd.read_csv(fetch(SAVANT.format(f"2024-06-{a}", f"2024-06-{b}"), os.path.join(D, f"pitches_2024-06-{a}.csv")),
                            usecols=["game_date", "game_pk", "at_bat_number", "pitch_number", "batter", "pitcher", "player_name", "balls", "strikes",
                                     "pitch_type", "release_speed", "description", "events", "launch_speed", "launch_angle", "hit_distance_sc"])
                for a, b in CHUNKS], ignore_index=True)
pt["home_run"] = (pt["events"] == "home_run").astype(int)
pt["count"] = pt["balls"].astype(str) + "-" + pt["strikes"].astype(str)
N, H = len(pt), int(pt["home_run"].sum())
by_count = pt.groupby("count")["home_run"].agg(["mean", "sum", "size"])
rule = pt["count"].isin(["3-1", "3-2"]).values.astype(int)       # 3볼이면 '홈런'
y = pt["home_run"].values
tp = int(((rule == 1) & (y == 1)).sum()); fp = int(((rule == 1) & (y == 0)).sum()); fn = H - tp
pa = pt.groupby(["game_pk", "at_bat_number"]).agg(hr=("home_run", "max"), batter=("batter", "first"))
bat = pa.groupby("batter")["hr"].max()
oh = pt[pt.batter == OHTANI]
oh_hr = oh[oh.home_run == 1].sort_values(["game_date", "at_bat_number"])
g, ab = oh_hr.iloc[0][["game_pk", "at_bat_number"]]                 # 6월 오타니의 첫 홈런 타석을 공 단위로
seq = pt[(pt.game_pk == g) & (pt.at_bat_number == ab)].sort_values("pitch_number")
J["pitch"] = dict(n=N, hr=H, rate=H / N, acc_zero=(N - H) / N,
                  count_rate={c: float(r) for c, r in by_count["mean"].items()}, count_n={c: int(v) for c, v in by_count["size"].items()},
                  rule_called=int(rule.sum()), tp=tp, fp=fp, fn=fn, acc_rule=(N - fp - fn) / N, precision=tp / rule.sum(), recall=tp / H,
                  n_pa=len(pa), pa_hr=int(pa.hr.sum()), pa_rate=float(pa.hr.mean()), pitches_per_pa=N / len(pa),
                  n_batters=len(bat), batters_hr=int(bat.sum()), batters_rate=float(bat.mean()),
                  ohtani_pitches=len(oh), ohtani_hr=int(oh.home_run.sum()),
                  example=dict(date=str(seq.game_date.iloc[0]), rows=[[int(r.pitch_number), f"{r.balls}-{r.strikes}", r.pitch_type,
                               round(r.release_speed * 1.609), r.description, (r.events if isinstance(r.events, str) else "")] for r in seq.itertuples()]))

# ---- 1. 던지기 전 두 문제: 같은 투구, 다른 y (스윙할까 · 구속)
SWING = {"hit_into_play", "foul", "swinging_strike", "swinging_strike_blocked", "foul_tip", "foul_bunt", "missed_bunt", "bunt_foul_tip"}
pp = pt.dropna(subset=["release_speed", "pitch_type"]).copy()
pp["swing"] = pp["description"].isin(SWING).astype(int)
pp["kmh"] = pp["release_speed"] * 1.609
IN = ["count", "batter", "pitcher"]
tr, te = pp[pp.game_date <= "2024-06-23"], pp[pp.game_date >= "2024-06-24"]
onehot = lambda cols: OneHotEncoder(handle_unknown="ignore")
clf = make_pipeline(onehot(IN), LogisticRegression(max_iter=2000)).fit(tr[IN].astype(str), tr.swing)
clf_c = make_pipeline(onehot(["count"]), LogisticRegression(max_iter=2000)).fit(tr[["count"]].astype(str), tr.swing)
reg = make_pipeline(onehot(IN), Ridge(alpha=1.0)).fit(tr[IN].astype(str), tr.kmh)
leak = make_pipeline(onehot(IN + ["pitch_type"]), Ridge(alpha=1.0)).fit(tr[IN + ["pitch_type"]].astype(str), tr.kmh)
J["pre"] = dict(n=len(pp), n_train=len(tr), n_test=len(te), swing_rate=float(pp.swing.mean()),
                acc_base=float(np.mean(te.swing == 0)), acc_count=float(np.mean(clf_c.predict(te[["count"]].astype(str)) == te.swing)),
                acc_model=float(np.mean(clf.predict(te[IN].astype(str)) == te.swing)),
                swing_by_count={c: float(r) for c, r in pp.groupby("count").swing.mean().items()},
                mean_kmh=float(tr.kmh.mean()), mae_base=mae(te.kmh, np.full(len(te), tr.kmh.mean())),
                mae_model=mae(te.kmh, reg.predict(te[IN].astype(str))), mae_leak=mae(te.kmh, leak.predict(te[IN + ["pitch_type"]].astype(str))))

os.makedirs(os.path.join(HERE, "assets"), exist_ok=True)
json.dump(J, open(os.path.join(HERE, "assets", "baseball.json"), "w"), indent=1, ensure_ascii=False, default=int)
print(json.dumps(J, indent=1, ensure_ascii=False, default=int))
