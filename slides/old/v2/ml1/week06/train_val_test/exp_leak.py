"""데이터 누수 실험: 의미 없는 난수 5,000개 + 동전 던지기 정답(앞/뒤) 100명.
누수: 전체 100명으로 정답과 가장 잘 맞는 숫자 20개를 먼저 고르고 → 5겹 교차검증.
정상: 숫자 고르기를 각 겹의 학습 데이터 안에서만 → 5겹 교차검증.  50번(시드 0~49) 반복 → assets/leak.json"""
import json, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
N, P, K, R = 100, 5000, 20, 50

def top(Xa, ya):
    c = np.abs(((Xa - Xa.mean(0)) * (ya - ya.mean())[:, None]).sum(0))
    return np.argsort(-c)[:K]

def run(seed):
    r = np.random.default_rng(seed); X = r.normal(size=(N, P)); y = np.array([0, 1] * (N // 2)); r.shuffle(y)
    sel = top(X, y); bad, ok = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        bad.append(LogisticRegression(max_iter=2000).fit(X[tr][:, sel], y[tr]).score(X[te][:, sel], y[te]))
        s2 = top(X[tr], y[tr])
        ok.append(LogisticRegression(max_iter=2000).fit(X[tr][:, s2], y[tr]).score(X[te][:, s2], y[te]))
    return float(np.mean(bad)), float(np.mean(ok))

res = [run(s) for s in range(R)]
out = dict(N=N, P=P, K=K, R=R, leak=[a for a, _ in res], ok=[b for _, b in res])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "leak.json"), "w"))
print(np.mean(out["leak"]), np.mean(out["ok"]), min(out["leak"]), max(out["ok"]))
