"""월드컵 골 수 데이터 만들기 (한 번만 실행). 실행: python prep_worldcup.py
원본: martj42/international_results (국가대표 경기 결과 1872~, CC0) results.csv 를 내려받아
모든 경기로 Elo 점수를 차례로 계산 (eloratings.net 방식: K = 월드컵 60 · 대륙컵 50 · 예선 40 · 기타 대회 30 · 친선 20,
골 차 배수 1 / 1.5 / (11+N)/8, 홈 팀 +100) 하고, 월드컵 본선 경기만 팀마다 한 줄로 남긴다.
data/worldcup_goals.csv: date, team, opponent, elo_diff (경기 전 내 Elo − 상대 Elo, 홈 +100 포함), goals (넣은 골)"""
import os, urllib.request
URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
raw = os.path.join(HERE, "data", "results.csv")
if not os.path.exists(raw):
    urllib.request.urlretrieve(URL, raw)
df = pd.read_csv(raw).dropna(subset=["home_score", "away_score"])
df = df.sort_values("date", kind="stable")
def K(t):
    if t == "FIFA World Cup": return 60
    if t in ("UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup", "Gold Cup", "Confederations Cup"): return 50
    if "qualification" in t: return 40
    if t == "Friendly": return 20
    return 30
R = {}; rows = []
for d in df.itertuples(index=False):
    h, a = R.get(d.home_team, 1500.0), R.get(d.away_team, 1500.0)
    adv = 0 if d.neutral else 100
    dr = h + adv - a
    we = 1 / (10 ** (-dr / 400) + 1)
    gh, ga = int(d.home_score), int(d.away_score); n = abs(gh - ga)
    G = 1 if n <= 1 else 1.5 if n == 2 else (11 + n) / 8
    w = 1 if gh > ga else 0.5 if gh == ga else 0
    if d.tournament == "FIFA World Cup":
        rows.append((d.date, d.home_team, d.away_team, round(dr), gh))
        rows.append((d.date, d.away_team, d.home_team, round(-dr), ga))
    k = K(d.tournament) * G
    R[d.home_team] = h + k * (w - we); R[d.away_team] = a - k * (w - we)
out = pd.DataFrame(rows, columns=["date", "team", "opponent", "elo_diff", "goals"])
out.to_csv(os.path.join(HERE, "data", "worldcup_goals.csv"), index=False); print(len(out))
