"""길쭉한 골짜기 f(x, y) = (x² + 50y²)/2 에서 경사하강 · 모멘텀 · Adam 의 실제 경로 → paths.npz
시작 (-4, 1.5), 최솟값 (0, 0). '도착' = 거리 0.05 이내. 학습률은 각 방법에서 발산하지 않는 범위로 고정."""
import numpy as np
A = np.array([1.0, 50.0]); g = lambda p: A * p; f = lambda p: 0.5 * (A * p * p).sum()
S = np.array([-4.0, 1.5]); N = 300
def gd(lr=0.09):
    p = S.copy(); P = [p.copy()]
    for _ in range(N): p = p - lr * g(p); P.append(p.copy())
    return np.array(P)
def mom(lr=0.03, b=0.9):
    p = S.copy(); v = np.zeros(2); P = [p.copy()]
    for _ in range(N): v = b * v + g(p); p = p - lr * v; P.append(p.copy())
    return np.array(P)
def adam(lr=0.3, b1=0.9, b2=0.999, eps=1e-8):
    p = S.copy(); m = np.zeros(2); s = np.zeros(2); P = [p.copy()]
    for t in range(1, N + 1):
        gr = g(p); m = b1 * m + (1 - b1) * gr; s = b2 * s + (1 - b2) * gr ** 2
        p = p - lr * (m / (1 - b1 ** t)) / (np.sqrt(s / (1 - b2 ** t)) + eps); P.append(p.copy())
    return np.array(P)
def steps(P):
    ok = np.linalg.norm(P, axis=1) < 0.05
    return int(np.argmax(ok)) if ok.any() else 10 ** 6
# 각 방법의 학습률을 격자에서 골라 공정하게 비교 (가장 빨리 도착하는 값)
best = {}
for k, fn, grid in (("gd", gd, np.linspace(0.005, 0.039, 35)), ("mom", mom, np.linspace(0.001, 0.039, 39)), ("adam", adam, np.linspace(0.02, 1.0, 50))):
    lr = min(grid, key=lambda l: steps(fn(lr=l))); best[k] = float(lr)
print("best lr", best)
R = {k: fn(lr=best[k]) for k, fn in (("gd", gd), ("mom", mom), ("adam", adam))}
arrive = {k: int(np.argmax(np.linalg.norm(P, axis=1) < 0.05)) if (np.linalg.norm(P, axis=1) < 0.05).any() else -1 for k, P in R.items()}
print(arrive)
import os; np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "paths.npz"), **R, arrive=[arrive[k] for k in ("gd", "mom", "adam")], lr=[best[k] for k in ("gd", "mom", "adam")])
