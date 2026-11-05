"""문제 정형화 덱의 야구 숫자 → assets/baseball.json. 실습 노트북 (lab/make_lab.py) 과 같은 데이터 · 같은 설정.
  타구 (Baseball Savant / Statcast, 2024-06 정규시즌 인플레이 타구):
     문제 A = 타구 속도 · 발사각 → 홈런인가 (분류, 정확도), 문제 B = 같은 입력 → 비거리 m (회귀, MAE)
     학습 70% · 시험 30% (random_state=0, 홈런 층화), 입력 = 속도 · 각도 · 각도² , 표준화 + 로지스틱회귀 / 선형회귀
     기준선 = 늘 '홈런 아님' / 학습 평균 비거리
  선수 시즌 (MLB Stats API, 2021~2024 정규시즌 타자 기록):
     다음 시즌 홈런 수. 학습 = 2021→2022 · 2022→2023, 시험 = 2023→2024, 두 해 모두 300타석 이상
     기준선 = 작년 그대로 · 학습 평균, 모델 = kNN (k=3, 작년 타석 · 타수 · 안타 · 홈런 · 볼넷 · 삼진 표준화) · 선형회귀 (작년 홈런)
  타석 (2024 정규시즌 182,449타석): 전부 '홈런 아님' vs 규칙 '2023년 30홈런 이상 타자면 홈런', 한 행 = 타석 vs 선수
실행: python3 exp_baseball.py  (처음엔 타구 CSV 받는 데 1분쯤)"""
import json, os, urllib.request
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "lab", "data"); os.makedirs(D, exist_ok=True)
UA = {"User-Agent": "book-ml-course"}
SAVANT = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details&player_type=batter"
          "&hfGT=R%7C&hfSea=2024%7C&hfPR=hit%5C.%5C.into%5C.%5C.play%7C&game_date_gt=2024-06-01&game_date_lt=2024-06-30")
STATS = "https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting&season={}&sportIds=1&playerPool=ALL&limit=3000&gameType=R"


def fetch(url, path):
    if not os.path.exists(path):
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


def season(year):
    js = json.load(open(fetch(STATS.format(year), os.path.join(D, f"hitting_{year}.json"))))
    rows = [dict(pid=s["player"]["id"], name=s["player"]["fullName"], **{k: s["stat"][v] for k, v in
            [("pa", "plateAppearances"), ("ab", "atBats"), ("h", "hits"), ("hr", "homeRuns"), ("bb", "baseOnBalls"), ("so", "strikeOuts")]})
            for s in js["stats"][0]["splits"]]
    return pd.DataFrame(rows).sort_values("pid").reset_index(drop=True)


J = {}
mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

# ---- 1. 타구: 같은 데이터, 두 문제
bb = pd.read_csv(fetch(SAVANT, os.path.join(D, "statcast_2024_06_in_play.csv")))
bb = bb[["game_date", "launch_speed", "launch_angle", "hit_distance_sc", "events"]].dropna().reset_index(drop=True)
bb["speed_kmh"] = bb["launch_speed"] * 1.609
bb["distance_m"] = bb["hit_distance_sc"] * 0.3048
bb["home_run"] = (bb["events"] == "home_run").astype(int)
X = np.c_[bb["speed_kmh"], bb["launch_angle"], bb["launch_angle"] ** 2]
y_a, y_b = bb["home_run"].values, bb["distance_m"].values
tr, te = train_test_split(np.arange(len(bb)), test_size=0.3, stratify=y_a, random_state=0)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y_a[tr])
pa_ = clf.predict(X[te])
reg = make_pipeline(StandardScaler(), LinearRegression()).fit(X[tr], y_b[tr])
J["batted"] = dict(n=len(bb), hr=int(y_a.sum()), n_test=len(te), hr_test=int(y_a[te].sum()),
                   acc_base=float(np.mean(y_a[te] == 0)), acc_model=float(np.mean(pa_ == y_a[te])),
                   model_caught=int(((pa_ == 1) & (y_a[te] == 1)).sum()), model_called=int(pa_.sum()),
                   mae_base=mae(y_b[te], np.full(len(te), y_b[tr].mean())), mae_model=mae(y_b[te], reg.predict(X[te])),
                   mean_dist=float(y_b[tr].mean()), max_dist=float(y_b.max()),
                   mae_leak=mae(y_b[te], make_pipeline(StandardScaler(), LinearRegression()).fit(
                       np.c_[X, y_b][tr], y_b[tr]).predict(np.c_[X, y_b][te])),
                   rows=bb.head(3)[["game_date", "speed_kmh", "launch_angle", "distance_m", "events"]].round(1).values.tolist())

# ---- 2. 선수 시즌: 다음 시즌 홈런 수
S = {y: season(y) for y in (2021, 2022, 2023, 2024)}
F = ["pa", "ab", "h", "hr", "bb", "so"]


def pairs(y0, min_pa=300):
    m = S[y0].merge(S[y0 + 1][["pid", "pa", "hr"]], on="pid", suffixes=("", "_next"))
    return m[(m.pa >= min_pa) & (m.pa_next >= min_pa)].reset_index(drop=True)


trp, tep = pd.concat([pairs(2021), pairs(2022)], ignore_index=True), pairs(2023)
yt = tep["hr_next"].values
sc = StandardScaler().fit(trp[F])
knn = KNeighborsRegressor(3).fit(sc.transform(trp[F]), trp["hr_next"])
lin = LinearRegression().fit(trp[["hr"]], trp["hr_next"])
J["next"] = dict(n_train=len(trp), n_test=len(tep), mae_mean=mae(yt, np.full(len(yt), trp["hr_next"].mean())),
                 mae_last=mae(yt, tep["hr"]), mae_knn=mae(yt, knn.predict(sc.transform(tep[F]))),
                 mae_lin=mae(yt, lin.predict(tep[["hr"]])), lin_coef=float(lin.coef_[0]), lin_b=float(lin.intercept_),
                 mae_leak=mae(yt, tep["hr"] / tep["pa"] * tep["pa_next"]))

# ---- 3. 타석: 정확도의 함정 · 정밀도 · 재현율 · 한 행
a, b = S[2023], S[2024]
N, P = int(b.pa.sum()), int(b.hr.sum())
slug = set(a.loc[a.hr >= 30, "pid"]); s = b[b.pid.isin(slug)]
tp, called = int(s.hr.sum()), int(s.pa.sum()); fp, fn = called - tp, P - tp
bat = b[b.pa > 0]
J["pa"] = dict(n=N, hr=P, rate=P / N, acc_zero=(N - P) / N, sluggers=len(slug), sluggers_2024=len(s), called=called,
               tp=tp, fp=fp, fn=fn, acc_rule=(N - fp - fn) / N, precision=tp / called, recall=tp / P,
               players=len(bat), players_hr=int((bat.hr > 0).sum()), players_rate=float((bat.hr > 0).mean()),
               top=b.sort_values("hr", ascending=False).head(3)[["name", "pa", "hr"]].values.tolist())

os.makedirs(os.path.join(HERE, "assets"), exist_ok=True)
json.dump(J, open(os.path.join(HERE, "assets", "baseball.json"), "w"), indent=1, ensure_ascii=False, default=int)
print(json.dumps(J, indent=1, ensure_ascii=False, default=int))
