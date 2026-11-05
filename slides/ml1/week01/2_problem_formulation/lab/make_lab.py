"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: Baseball Savant (2024-06 인플레이 타구 CSV) · MLB Stats API (2021~2024 타자 기록) · MovieLens (GroupLens).
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
- **파트 2**: 다음 시즌 홈런 수: 기준선 (작년 그대로) vs kNN · 선형회귀, 시간 순서로 나누기
- **파트 3**: 2024년 모든 타석: 정확도의 함정, 정밀도 · 재현율, 한 행을 타석 → 타자로
- **파트 4**: 넷플릭스 프라이즈 축소판: 사람마다 최근 별점을 숨기고 RMSE, '10% 개선' 목표

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요. 코드 칸 아래의 **예상 결과** 와 비교하면 맞았는지 바로 알 수 있습니다."""),
("code", """import io, json, os, zipfile, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
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
("md", """2024년 6월 MLB 정규시즌에서 **인플레이로 날아간 타구** 를 Baseball Savant (MLB Statcast) 에서 받는다. 한 행 = 타구 하나. 로그인 없이 받을 수 있고, 처음엔 1분쯤 걸린다 (약 17MB).

- `launch_speed`: 타구 속도 (mph → km/h 로 바꾼다), `launch_angle`: 발사각 (도)
- `hit_distance_sc`: 비거리 (피트 → m 로 바꾼다), `events`: 결과 (`home_run`, `single`, `field_out` …)"""),
("code", """os.makedirs("data", exist_ok=True)
SAVANT = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details&player_type=batter"
          "&hfGT=R%7C&hfSea=2024%7C&hfPR=hit%5C.%5C.into%5C.%5C.play%7C&game_date_gt=2024-06-01&game_date_lt=2024-06-30")
PATH = "data/statcast_2024_06_in_play.csv"
if not os.path.exists(PATH):
    req = urllib.request.Request(SAVANT, headers={"User-Agent": "book-ml-course"})
    with urllib.request.urlopen(req, timeout=300) as r, open(PATH, "wb") as f:
        f.write(r.read())

df = pd.read_csv(PATH)[["game_date", "launch_speed", "launch_angle", "hit_distance_sc", "events"]].dropna().reset_index(drop=True)
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
("md", """**생각해 보기**: 문제 A 의 정확도 (96.7%) 와 기준선 (95.5%) 의 차이는 1.2%p 뿐이다. 그런데 모델은 시험의 홈런 275개 중 몇 개를 맞혔을까? `clf.predict(X[idx_te])` 로 세어 보자. 그리고 문제 B 의 입력에 `distance_m` 을 넣으면 MAE 가 얼마가 되는지, 그 점수가 왜 의미 없는지 말해 보자."""),

# ---------------- 파트 2. 다음 시즌 홈런 수
("md", """## 파트 2. 다음 시즌 홈런 수: 기준선이 모델을 이길 수 있다

MLB 공식 기록 (MLB Stats API) 에서 2021~2024 정규시즌 타자 기록을 받는다. 한 행 = 타자의 한 시즌. 올해 기록으로 **내년 홈런 수** 를 맞힌다.

**시간 순서대로** 나눈다: 2021→2022, 2022→2023 으로 학습, 2023→2024 로 시험. 두 해 모두 300타석 이상인 타자만 쓴다."""),
("code", """STATS = "https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting&season={}&sportIds=1&playerPool=ALL&limit=3000&gameType=R"
COLS = {"pa": "plateAppearances", "ab": "atBats", "h": "hits", "hr": "homeRuns", "bb": "baseOnBalls", "so": "strikeOuts"}

def load_season(year):
    path = f"data/hitting_{year}.json"
    if not os.path.exists(path):
        req = urllib.request.Request(STATS.format(year), headers={"User-Agent": "book-ml-course"})
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    splits = json.load(open(path))["stats"][0]["splits"]
    rows = [dict(pid=s["player"]["id"], name=s["player"]["fullName"], **{k: s["stat"][v] for k, v in COLS.items()}) for s in splits]
    return pd.DataFrame(rows).sort_values("pid").reset_index(drop=True)

seasons = {year: load_season(year) for year in (2021, 2022, 2023, 2024)}
for year, s in seasons.items():
    print(year, "batters:", len(s), " PA:", s["pa"].sum(), " HR:", s["hr"].sum())"""),
("code", """def make_pairs(this_year, next_year, min_pa=300):
    \"\"\"One row per batter: this year's stats + next year's PA and HR (both years >= min_pa)\"\"\"
    m = this_year.merge(next_year[["pid", "pa", "hr"]], on="pid", suffixes=("", "_next"))
    return m[(m["pa"] >= min_pa) & (m["pa_next"] >= min_pa)].reset_index(drop=True)

train = pd.concat([make_pairs(seasons[2021], seasons[2022]), make_pairs(seasons[2022], seasons[2023])], ignore_index=True)
test = make_pairs(seasons[2023], seasons[2024])
print("train", len(train), " test", len(test))
test[["name", "pa", "hr", "hr_next"]].sort_values("hr", ascending=False).head(5)"""),
("md", """기준선 두 개와 모델 두 개를 비교한다.
- 기준선 1: 늘 학습 데이터의 평균 (내년 홈런 수)
- 기준선 2: **작년 그대로** (올해 홈런 수를 내년에도 친다)
- kNN 모델 (k = 3): 올해 기록 (타석 · 타수 · 안타 · 홈런 · 볼넷 · 삼진) 이 비슷한 타자 3명의 내년 홈런 평균
- 선형회귀: 올해 홈런 하나만 입력"""),
("code", """FEATURES = ["pa", "ab", "h", "hr", "bb", "so"]
y_test = test["hr_next"].values

pred_mean = np.full(len(test), train["hr_next"].mean())
pred_last = test["hr"].values      #@ANS pred_last = None   # TODO: this year's HR, unchanged

scaler = StandardScaler().fit(train[FEATURES])
knn = KNeighborsRegressor(3).fit(scaler.transform(train[FEATURES]), train["hr_next"])
pred_knn = knn.predict(scaler.transform(test[FEATURES]))
lin = LinearRegression().fit(train[["hr"]], train["hr_next"])
pred_lin = lin.predict(test[["hr"]])

results = {"training mean": pred_mean, "same as last year": pred_last, "kNN (k=3)": pred_knn, "linear (HR only)": pred_lin}
for name, pred in results.items():
    print(f"{name:18s} MAE {mae(y_test, pred):.2f}")
print(f"linear: next = {lin.coef_[0]:.2f} x this + {lin.intercept_:.1f}")
plt.figure(figsize=(7, 3)); plt.bar(results.keys(), [mae(y_test, p) for p in results.values()], color=["gray", "navy", "red", "green"])
plt.ylabel("test MAE (home runs)"); plt.xticks(fontsize=8); plt.show()"""),
("md", """**관찰**: kNN 모델 (6.65) 이 '작년 그대로' (6.42) 보다 못하다. 기준선을 재 보지 않았다면 몰랐을 일이다. 선형회귀는 작년 홈런에 0.61 을 곱하고 6 을 더한다. 많이 친 타자는 조금 덜, 적게 친 타자는 조금 더 칠 거라고 **평균 쪽으로 당긴다**.

**생각해 보기**: 예측에 '올해 홈런 비율 × **내년 실제 타석 수**' 를 쓰면 MAE 가 더 좋아진다 (해 보자). 그런데 왜 이 숫자는 믿으면 안 될까?"""),

# ---------------- 파트 3. 타석
("md", """## 파트 3. 2024년 모든 타석: 정확도의 함정

파트 2 에서 받은 2024 기록으로 **한 행 = 타석 하나** 인 데이터를 만든다. 타자마다 타석 수만큼 행을 만들고, 홈런 수만큼은 `y = 1`, 나머지는 `y = 0` 이다. (어느 타석이 홈런이었는지 순서는 모르지만, 개수를 세는 지표에는 상관없다.)"""),
("code", """def plate_appearances(season):
    \"\"\"One row per plate appearance: batter id and y (1 = home run)\"\"\"
    s = season[season["pa"] > 0]
    pid = np.repeat(s["pid"].values, s["pa"].values)
    y = np.concatenate([np.r_[np.ones(hr, dtype=int), np.zeros(pa - hr, dtype=int)] for pa, hr in zip(s["pa"], s["hr"])])
    return pid, y

pa_pid, y = plate_appearances(seasons[2024])
print("plate appearances:", len(y), "  home runs:", y.sum(), f"({y.mean():.2%})")

all_no = np.zeros(len(y), dtype=int)      #@ANS all_no = None   # TODO: predict 0 (no HR) for every plate appearance
print(f"'always no HR' accuracy: {accuracy(y, all_no):.2%}")"""),
("md", """정확도가 매우 높지만 홈런을 하나도 못 잡는다. 규칙 하나를 만들어 보자: **2023년에 30홈런 이상 친 타자가 서면 '홈런'**."""),
("code", """s23 = seasons[2023]
sluggers = s23.loc[s23["hr"] >= 30, "pid"]
print("2023 sluggers (30+ HR):", len(sluggers))
rule = np.isin(pa_pid, sluggers).astype(int)      #@ANS rule = None   # TODO: 1 if the batter is a slugger, else 0
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

for name, pred in [("always no HR", all_no), ("rule: 30+ HR last year", rule)]:
    tp, fp, fn, p, r = precision_recall(y, pred)
    print(f"{name:24s} caught {tp:4d}  false alarms {fp:6d}  missed {fn:5d}  precision {p:.2%}  recall {r:.1%}")"""),
("md", """### 한 행 = 타자 한 명

이번엔 타자 한 명 (2024 시즌) 을 한 행으로, **홈런을 한 번이라도 쳤으면 1** 로 정해 보자."""),
("code", """s24 = seasons[2024][seasons[2024]["pa"] > 0]
batter_y = (s24["hr"] > 0).astype(int)      #@ANS batter_y = None   # TODO: 1 if the batter hit at least one HR
print("batters:", len(batter_y), "  with a HR:", batter_y.sum(), f"({batter_y.mean():.1%})")
print(f"plate-appearance rate {y.mean():.2%}  ->  batter rate {batter_y.mean():.1%}")"""),
("md", """**관찰**: 같은 기록인데 한 행을 무엇으로 정하느냐에 따라 '홈런 비율' 이 2.99% 에서 80.2% 로 바뀐다.

**생각해 보기**: 타석에 서기 **전에** 예측한다면 볼카운트 (2볼 2스트라이크 …) 를 입력에 넣을 수 있을까? 한 행을 **투구 하나** 로 바꾸면 어떻게 될까?"""),

# ---------------- 파트 4. 넷플릭스 프라이즈 축소판
("md", """## 파트 4. 넷플릭스 프라이즈 축소판: 영화 별점

넷플릭스 프라이즈 (2006) 는 '사용자마다 **가장 최근** 별점을 숨기고, RMSE 로 채점, 기준선보다 10% 나으면 우승' 이었다. 같은 규칙을 MovieLens (610명 · 별점 10만 개) 로 작게 해 보자."""),
("code", """URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/ml-latest-small.zip"):
    urllib.request.urlretrieve(URL, "data/ml-latest-small.zip")
with zipfile.ZipFile("data/ml-latest-small.zip") as z:
    ratings = pd.read_csv(z.open("ml-latest-small/ratings.csv"))
print(ratings.shape, "users", ratings["userId"].nunique(), "movies", ratings["movieId"].nunique())

# hide each user's most recent 10% of ratings (like the Netflix Prize qualifying set)
ratings = ratings.sort_values(["userId", "timestamp"])
rank = ratings.groupby("userId").cumcount(ascending=False)            # 0 = newest
n_user = ratings.groupby("userId")["rating"].transform("size")
hidden = rank < np.maximum(1, (n_user * 0.1).astype(int))
train, test = ratings[~hidden], ratings[hidden]
print("train", len(train), "hidden test", len(test))"""),
("md", """세 가지 예측의 RMSE 를 비교한다.
- 기준선 1: 늘 전체 평균 별점
- 기준선 2: 영화마다 평균 별점 (넷플릭스 이전의 단순한 방법)
- 모델: 평균 + 사용자 치우침 + 영화 치우침 (후하게 주는 사람, 다들 좋아하는 영화)

그리고 '영화 평균보다 10% 나은가?' 를 확인한다."""),
("code", """def fit_biases(train, reg=5.0):
    \"\"\"Global mean + movie bias + user bias, shrunk toward 0 for movies/users with few ratings\"\"\"
    mu = train["rating"].mean()
    r = train.assign(res=train["rating"] - mu)
    b_movie = r.groupby("movieId")["res"].agg(lambda v: v.sum() / (len(v) + reg))
    r = r.assign(res=r["res"] - r["movieId"].map(b_movie))
    b_user = r.groupby("userId")["res"].agg(lambda v: v.sum() / (len(v) + reg))
    return mu, b_user, b_movie

mu, b_user, b_movie = fit_biases(train)
movie_mean = train.groupby("movieId")["rating"].mean()
preds = {"global mean": np.full(len(test), mu),
         "movie mean": test["movieId"].map(movie_mean).fillna(mu).values,
         "mean + user + movie bias": (mu + test["userId"].map(b_user).fillna(0) + test["movieId"].map(b_movie).fillna(0)).clip(0.5, 5).values}
scores = {k: rmse(test["rating"].values, v) for k, v in preds.items()}      #@ANS scores = None   # TODO: RMSE of each prediction (use your rmse function)
for k, v in scores.items():
    print(f"{k:26s} RMSE {v:.4f}")
goal = scores["movie mean"] * 0.9
print(f"10% better than movie mean = {goal:.4f} -> reached: {scores['mean + user + movie bias'] < goal}")"""),
("md", """**관찰**: '영화마다 평균' 은 전체 평균보다 조금 낫다. 여기에 사람마다 후하게 · 짜게 주는 치우침을 더하면 RMSE 가 크게 내려가 '10% 개선' 을 넘긴다. 진짜 넷플릭스 데이터에서는 기준선 (Cinematch) 이 훨씬 강해서, 우승 팀은 100개가 넘는 모델을 섞어 3년 만에 10% 를 넘겼다.

**확인 질문**
1. 파트 1 문제 A 에서 정확도가 기준선보다 1.2%p 높다. 이 모델은 쓸 만한가? 홈런을 몇 개 잡았는지로 말해 보자.
2. 파트 2 에서 kNN 이 '작년 그대로' 에 진 이유는 무엇일까? (힌트: 비슷한 타자 3명의 내년은 얼마나 들쭉날쭉할까?)
3. 파트 3 에서 '전부 홈런' 이라고 답하면 재현율과 정밀도는 각각 얼마가 될까?
4. 파트 2 의 시험 데이터를 2023→2024 대신 모든 해를 섞어 무작위로 뽑으면 무엇이 문제일까?
5. 파트 4 에서 숨길 별점을 '사람마다 가장 최근' 이 아니라 무작위로 고르면 RMSE 가 어떻게 바뀔까? 어느 쪽이 실제 서비스와 닮았을까?

데이터 출처: Baseball Savant (MLB Statcast) · MLB Stats API (statsapi.mlb.com) · MovieLens (Harper & Konstan, 2015) · Netflix Prize (Bennett & Lanning, 2007)."""),
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
