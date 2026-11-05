"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: Baseball Savant (2024-06 모든 투구 CSV, 타구는 그중 인플레이).
슬라이드 숫자 (../exp_baseball.py) 와 같은 데이터 · 같은 설정."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = __import__("re").sub(r"^\d+_", "", os.path.basename(os.path.dirname(HERE)))          # 폴더 이름 (problem_formulation) 이 파일명 앞에 붙는다

CELLS = [
("md", """# 문제 정형화와 평가지표 실습: 문제를 직접 정해 보기

모델보다 먼저 **다섯 칸** 을 채운다: ① 한 행은 무엇? ② 입력 X ③ 출력 y ④ 평가지표 ⑤ 기준선.

- **파트 1**: 타구: 같은 데이터로 두 문제 (홈런인가 분류 · 비거리 회귀), 정확도 · MAE · 기준선 직접 만들기
- **파트 2**: 6월의 모든 공: 정확도의 함정, 볼카운트 규칙의 정밀도 · 재현율, 한 행을 공 → 타석 → 타자로

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요. 코드 칸 아래의 **예상 결과** 와 비교하면 맞았는지 바로 알 수 있습니다."""),
("code", """import json, os, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler"""),

# ---------------- 파트 1. 타구
("md", """## 파트 1. 타구: 같은 데이터, 두 문제

먼저 채점 함수 세 개를 만든다. 정답 `y_true` 와 예측 `y_pred` 를 받는다.
- 정확도 = 맞힌 개수 ÷ 전체
- MAE = 오차 절댓값의 평균
- RMSE = 오차 제곱의 평균에 루트"""),
("code", """def accuracy(y_true, y_pred):
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))      #@ANS return None   # TODO: fraction of exact matches

def mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))      #@ANS return None   # TODO: mean of |error|

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))      #@ANS return None   # TODO: sqrt of mean of error^2

print("accuracy:", accuracy([0, 1, 2, 2], [0, 1, 1, 2]))          # 0.75
print("MAE :", mae([0, 0, 0, 0], [1, 1, 1, 9]))                  # slide: 3
print("RMSE:", round(rmse([0, 0, 0, 0], [1, 1, 1, 9]), 2))       # slide: 4.58"""),
("md", """2024년 6월 MLB 정규시즌에 던진 **모든 공** 을 Baseball Savant (MLB Statcast) 에서 받는다. 한 행 = 공 하나 (투구). 한 번에 25,000줄까지만 주니 5일씩 6번 받는다. 로그인 없이 받을 수 있고, 처음엔 4~5분 걸린다. 파트 1 과 파트 2 가 이 데이터를 같이 쓴다.

- `balls`, `strikes`: **던지기 직전** 볼카운트, `pitch_type`, `release_speed`: 구종 · 구속 (mph)
- `description`: 그 공의 결과 (`ball`, `swinging_strike`, `hit_into_play` …), `events`: 타석이 끝났으면 그 결과 (`home_run`, `single`, `field_out` …)
- 인플레이 타구에는 `launch_speed` (타구 속도, mph), `launch_angle` (발사각), `hit_distance_sc` (비거리, 피트) 가 있다"""),
("code", """os.makedirs("data", exist_ok=True)
SAVANT = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details&player_type=batter"
          "&hfGT=R%7C&hfSea=2024%7C&game_date_gt={}&game_date_lt={}")
CHUNKS = [("01", "05"), ("06", "10"), ("11", "15"), ("16", "20"), ("21", "25"), ("26", "30")]
COLS = ["game_date", "game_pk", "at_bat_number", "pitch_number", "batter", "player_name", "balls", "strikes",
        "pitch_type", "release_speed", "description", "events", "launch_speed", "launch_angle", "hit_distance_sc"]

def load_pitches(chunks):
    parts = []
    for a, b in chunks:
        path = f"data/pitches_2024-06-{a}.csv"
        if not os.path.exists(path):
            req = urllib.request.Request(SAVANT.format(f"2024-06-{a}", f"2024-06-{b}"), headers={"User-Agent": "book-ml-course"})
            with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
                f.write(r.read())
        parts.append(pd.read_csv(path, usecols=COLS))
    return pd.concat(parts, ignore_index=True)

pitches = load_pitches(CHUNKS)
pitches["home_run"] = (pitches["events"] == "home_run").astype(int)
print(pitches.shape, " home runs:", pitches["home_run"].sum())
pitches[["game_date", "player_name", "balls", "strikes", "pitch_type", "description", "events"]].head(3)"""),
("md", """파트 1 에서는 그중 **인플레이로 날아간 타구** (`description == "hit_into_play"`) 만 쓴다. 한 행 = 타구 하나. 속도는 km/h, 비거리는 m 로 바꾼다."""),
("code", """df = pitches[pitches["description"] == "hit_into_play"][["game_date", "launch_speed", "launch_angle", "hit_distance_sc", "events"]].dropna().reset_index(drop=True)
df["speed_kmh"] = df["launch_speed"] * 1.609
df["distance_m"] = df["hit_distance_sc"] * 0.3048
df["home_run"] = (df["events"] == "home_run").astype(int)
print(df.shape, " home runs:", df["home_run"].sum())
df.head(3)"""),
("md", """### 문제 A: 타구 속도 · 발사각 → 홈런인가 (분류)

공이 방망이에 맞은 **순간** 알 수 있는 것만 입력으로 쓴다: 속도, 각도, 그리고 각도². (너무 낮게 뜨거나 너무 높이 뜨면 홈런이 안 되니, 각도의 제곱을 넣어 '가운데가 좋은' 모양을 만들 수 있게 한다.)

학습 70% · 시험 30% (홈런 비율을 맞춰서). **기준선** = 늘 '홈런 아님' 이라고 답하기."""),
("code", """X = np.c_[df["speed_kmh"], df["launch_angle"], df["launch_angle"] ** 2]      #@ANS X = None   # TODO: columns speed_kmh, launch_angle, launch_angle**2
y_a = df["home_run"].values                                                  #@ANS y_a = None   # TODO: the home_run column (0/1)
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.3, stratify=y_a, random_state=0)

base_pred = np.zeros(len(idx_te), dtype=int)          # always 'not a home run'
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[idx_tr], y_a[idx_tr])
print(f"baseline (always 'no HR'): accuracy {accuracy(y_a[idx_te], base_pred):.1%}")
print(f"logistic regression      : accuracy {accuracy(y_a[idx_te], clf.predict(X[idx_te])):.1%}")"""),
("md", """### 문제 B: 같은 타구, 같은 입력 → 비거리 (회귀)

이번엔 `y` 가 비거리 (m). 입력은 문제 A 와 같다. **비거리는 입력에 넣으면 안 된다** (답을 입력에 넣는 셈이다). 같은 행으로 학습 · 시험을 나눈다. **기준선** = 학습 데이터의 평균 비거리를 늘 답하기."""),
("code", """y_b = df["distance_m"].values      #@ANS y_b = None   # TODO: the distance_m column

base_pred = np.full(len(idx_te), y_b[idx_tr].mean())
reg = make_pipeline(StandardScaler(), LinearRegression()).fit(X[idx_tr], y_b[idx_tr])
print(f"baseline (always {y_b[idx_tr].mean():.1f} m): MAE {mae(y_b[idx_te], base_pred):.1f} m  RMSE {rmse(y_b[idx_te], base_pred):.1f} m")
print(f"linear regression        : MAE {mae(y_b[idx_te], reg.predict(X[idx_te])):.1f} m  RMSE {rmse(y_b[idx_te], reg.predict(X[idx_te])):.1f} m")"""),
("md", """**생각해 보기**: 문제 A 의 정확도 (96.8%) 와 기준선 (95.5%) 의 차이는 1.3%p 뿐이다. 그런데 모델은 시험의 홈런 275개 중 몇 개를 맞혔을까? `clf.predict(X[idx_te])` 로 세어 보자. 그리고 문제 B 의 입력에 `distance_m` 을 넣으면 MAE 가 얼마가 되는지, 그 점수가 왜 의미 없는지 말해 보자."""),

# ---------------- 파트 2. 공 하나 = 한 행
("md", """## 파트 2. 6월의 모든 공: 정확도의 함정

파트 1 에서 받은 `pitches` 를 그대로 쓴다. **한 행 = 공 하나**, 출력 `y` = 이 공이 홈런이 되었나 (0/1). 먼저 한 타석이 공 여러 개로 이루어진다는 걸 보자: 6월 오타니의 첫 홈런 타석을 공 단위로 펼친다."""),
("code", """OHTANI = 660271
oh_hr = pitches[(pitches["batter"] == OHTANI) & (pitches["home_run"] == 1)].sort_values(["game_date", "at_bat_number"])
game, at_bat = oh_hr.iloc[0][["game_pk", "at_bat_number"]]
one_pa = pitches[(pitches["game_pk"] == game) & (pitches["at_bat_number"] == at_bat)].sort_values("pitch_number")
one_pa[["game_date", "pitch_number", "balls", "strikes", "pitch_type", "release_speed", "description", "events", "home_run"]]"""),
("code", """y = pitches["home_run"].values
print("pitches:", len(y), "  home runs:", y.sum(), f"({y.mean():.2%})")

all_no = np.zeros(len(y), dtype=int)      #@ANS all_no = None   # TODO: predict 0 (no HR) for every pitch
print(f"'always no HR' accuracy: {accuracy(y, all_no):.2%}")"""),
("md", """정확도가 매우 높지만 홈런을 하나도 못 잡는다. **던지기 직전 볼카운트** 는 예측하는 순간 알 수 있는 입력이다. 볼카운트별로 홈런이 된 공의 비율을 보자."""),
("code", """pitches["count"] = pitches["balls"].astype(str) + "-" + pitches["strikes"].astype(str)
by_count = pitches.groupby("count")["home_run"].agg(["mean", "sum", "size"]).sort_values("mean")
print(by_count)
plt.figure(figsize=(7, 3)); plt.bar(by_count.index, by_count["mean"] * 100, color="gray")
plt.axhline(y.mean() * 100, color="navy", ls=":"); plt.xlabel("count before the pitch (balls-strikes)"); plt.ylabel("HR on this pitch (%)"); plt.show()"""),
("md", """규칙 하나를 만들어 보자: **볼카운트가 3-1 또는 3-2 이면 '홈런'**."""),
("code", """rule = pitches["count"].isin(["3-1", "3-2"]).values.astype(int)      #@ANS rule = None   # TODO: 1 if the count is 3-1 or 3-2, else 0
print(f"rule accuracy: {accuracy(y, rule):.2%}")"""),
("md", """### 정밀도와 재현율: 세는 법을 바꾼다

- **재현율** = 잡았다 ÷ 실제 홈런 전부 (실제 홈런 중 몇 개를 잡았나)
- **정밀도** = 잡았다 ÷ '홈런' 이라고 한 것 전부 (찍은 것 중 진짜는 몇 개)

(3주차 정밀도와 재현율에서 자세히 다룬다.)"""),
("code", """def precision_recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))     # caught
    fp = np.sum((y_pred == 1) & (y_true == 0))     # false alarm
    fn = np.sum((y_pred == 0) & (y_true == 1))     # missed
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0      #@ANS precision = None   # TODO: caught / everything we called HR
    recall = tp / (tp + fn)                                 #@ANS recall = None      # TODO: caught / all real HRs
    return tp, fp, fn, precision, recall

for name, pred in [("always no HR", all_no), ("rule: count 3-1 / 3-2", rule)]:
    tp, fp, fn, p, r = precision_recall(y, pred)
    print(f"{name:22s} caught {tp:4d}  false alarms {fp:6d}  missed {fn:4d}  precision {p:.2%}  recall {r:.1%}")"""),
("md", """### 한 행 = 공 → 타석 → 타자

같은 6월 기록으로 한 행을 바꿔 보자. 타석은 `game_pk` + `at_bat_number` 로 묶고 **홈런으로 끝났으면 1**, 타자는 `batter` 로 묶고 **6월에 홈런을 한 번이라도 쳤으면 1**."""),
("code", """pa_y = pitches.groupby(["game_pk", "at_bat_number"])["home_run"].max()      #@ANS pa_y = None   # TODO: per plate appearance (game_pk, at_bat_number), 1 if any pitch was a HR
batter_y = pitches.groupby("batter")["home_run"].max()                       #@ANS batter_y = None   # TODO: per batter, 1 if they hit any HR in June
print(f"pitch : {len(y):6d} rows  HR rate {y.mean():.2%}")
print(f"PA    : {len(pa_y):6d} rows  HR rate {pa_y.mean():.2%}   ({len(y) / len(pa_y):.2f} pitches per PA)")
print(f"batter: {len(batter_y):6d} rows  HR rate {batter_y.mean():.1%}")"""),
("md", """**관찰**: 같은 기록인데 한 행을 무엇으로 정하느냐에 따라 '홈런 비율' 이 0.79% → 3.06% → 64.6% 로 바뀐다. 볼카운트를 입력으로 쓸 수 있는 것도 한 행이 공 하나일 때뿐이다 (타석이 시작될 땐 늘 0-0).

**생각해 보기**: 한 행 = 타석 하나로 '이 타석이 홈런으로 끝날까?' 를 타석 시작 전에 맞힌다면, 입력으로 쓸 수 있는 것은 무엇일까?"""),
("md", """**확인 질문**
1. 파트 1 문제 A 에서 정확도가 기준선보다 1.3%p 높다. 이 모델은 쓸 만한가? 홈런을 몇 개 잡았는지로 말해 보자.
2. 파트 2 에서 모든 공에 '홈런' 이라고 답하면 재현율과 정밀도는 각각 얼마가 될까?
3. 파트 2 의 볼카운트 규칙에 '작년 30홈런 이상 타자' 같은 조건을 더하면 정밀도와 재현율은 어떻게 바뀔까?
4. 공 하나를 한 행으로 학습 · 시험을 나눌 때, 무작위로 섞으면 무엇이 문제일까? (힌트: 같은 타석의 바로 앞뒤 공)

데이터 출처: Baseball Savant (MLB Statcast)."""),
]

if __name__ == "__main__":
    REL = os.path.relpath(HERE, os.path.expanduser("~/book-ml"))
    COLAB = f"https://colab.research.google.com/github/SuminHan/book-ml/blob/main/{REL}/{TOPIC}.ipynb"
    kind, first = CELLS[0]                                   # 첫 칸 제목 아래에 Colab 배지
    CELLS[0] = (kind, first.replace("\n", f"\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})\n", 1))
    lab.build(CELLS, f"{HERE}/{TOPIC}.ipynb", f"{HERE}/{TOPIC}_solution.ipynb")
    sys.path.insert(0, os.path.join(HERE, "../../../../tools"))
    from expected import add_expected                         # 학생용 코드 칸 아래에 '예상 결과' (정답을 실제 실행한 출력)
    print(add_expected(f"{HERE}/{TOPIC}.ipynb", f"{HERE}/{TOPIC}_solution.ipynb"))
