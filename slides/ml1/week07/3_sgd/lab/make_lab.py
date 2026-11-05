"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: diabetes (scikit-learn 내장, 442명) · MNIST (torchvision 으로 내려받기, 학습 6만 장 · 시험 1만 장)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (sgd) 이 파일명 앞에 붙는다

CELLS = [
("md", """# 확률적 경사하강법(SGD) 실습

전체 데이터 대신 일부(미니배치)만 보고 기울기를 어림해 걷는다. 슬라이드의 내용을 진짜 데이터로 확인한다.

- **파트 1**: 미니배치 기울기 1,000개의 평균 = 전체 기울기? 배치 크기별 흩어짐
- **파트 2**: SGD 직접 구현 (섞기 · 자르기 · 걷기) 후 전체 배치와 비교
- **파트 3**: PyTorch 로 MNIST 54% vs 96% 재현 · 배치 크기 바꿔 보기

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes"""),
("md", """## 데이터: 당뇨병 진행도

환자 442명, 특징 10개. 비용 $J$ 와 전체 기울기 함수도 여기서 정의한다."""),
("code", """d = load_diabetes()
X = (d.data - d.data.mean(0)) / d.data.std(0)           # standardize 10 features
y = (d.target - d.target.mean()) / d.target.std()       # disease progression one year later (standardized)
m, n = X.shape
print(f"{m} patients, {n} features:", list(d.feature_names))

def cost(w, b, X, y):
    \"\"\"J = (1/m) Σ ½ (ŷ_i − y_i)²\"\"\"
    return 0.5 * np.mean((X @ w + b - y) ** 2)

def full_grad(w, b, X, y):
    \"\"\"full gradient = average of the per-example gradients\"\"\"
    r = X @ w + b - y                                    # residual y_hat - y
    return X.T @ r / len(y), r.mean()"""),
("md", """## 파트 1. 미니배치 기울기는 평균적으로 정확할까?

미니배치 $\\mathcal B$ ($B$명) 의 기울기: $g_{\\mathcal B} = \\dfrac{1}{B}\\sum_{i\\in\\mathcal B} g_i$. 슬라이드의 결과 $\\mathbb E[g_{\\mathcal B}] = \\partial J/\\partial w$ 를 확인한다."""),
("code", """def minibatch_grad(w, b, X, y, idx):
    \"\"\"idx: indices of the sampled examples. Same computation as full_grad, on those examples only\"\"\"
    r = X[idx] @ w + b - y[idx]                         #@ANS r = None                # TODO: residual of the sampled examples
    return X[idx].T @ r / len(idx), r.mean()

w0, b0 = np.zeros(n), 0.0
gw_full, gb_full = full_grad(w0, b0, X, y)
rng = np.random.default_rng(0)
for B in [1, 16, 64]:
    gs = np.array([minibatch_grad(w0, b0, X, y, rng.choice(m, B, replace=False))[0] for _ in range(1000)])
    cos = gs @ gw_full / (np.linalg.norm(gs, axis=1) * np.linalg.norm(gw_full))
    print(f"B={B:3d}  |mean - full gradient| {np.linalg.norm(gs.mean(0) - gw_full):.3f}   "
          f"spread (sum of std) {gs.std(0).sum():.2f}   points downhill {np.mean(cos > 0):.0%}")
print("full gradient norm", round(np.linalg.norm(gw_full), 3))"""),
("md", """**관찰**: 배치 크기와 상관없이 1,000개의 평균은 전체 기울기와 거의 같다. 다른 것은 흩어짐이다. $B$ 가 커질수록 덜 흩어지고, 내리막 쪽을 가리키는 비율이 올라간다."""),
("md", """## 파트 2. SGD 직접 구현

에폭마다 ① 섞고 ② 앞에서부터 $B$ 개씩 잘라 ③ 미니배치 기울기로 한 걸음."""),
("code", """def sgd(X, y, B, alpha, epochs, seed=0):
    m, n = X.shape
    rng = np.random.default_rng(seed)
    w, b = np.zeros(n), 0.0
    history = [cost(w, b, X, y)]
    for ep in range(epochs):
        perm = rng.permutation(m)                        #@ANS perm = None             # TODO: (1) shuffled order of 0..m-1
        for i in range(0, m, B):
            idx = perm[i:i + B]                          #@ANS idx = None              # TODO: (2) B examples starting at i
            gw, gb = minibatch_grad(w, b, X, y, idx)
            w, b = w - alpha * gw, b - alpha * gb        #@ANS w, b = None, None       # TODO: (3) one step
        history.append(cost(w, b, X, y))
    return w, b, history

alpha, epochs = 0.02, 20
plt.figure(figsize=(7, 4))
for B in [1, 16, m]:
    w, b, h = sgd(X, y, B, alpha, epochs)
    plt.plot(h, marker="o", ms=3, label=f"B={B} ({-(-m // B)} steps per epoch)")
    print(f"B={B:3d}: J after {epochs} epochs = {h[-1]:.3f}")
w_ls = np.linalg.lstsq(np.c_[X, np.ones(m)], y, rcond=None)[0]
print("minimum J =", round(cost(w_ls[:-1], w_ls[-1], X, y), 3))
plt.xlabel("epoch"); plt.ylabel("cost J"); plt.legend(); plt.title(f"same learning rate {alpha}"); plt.show()"""),
("md", """**해 보기**: `alpha` 를 0.2, 1.0 으로 바꿔 보자. 전체 배치(`B=m`)는 학습률을 키우면 빨라지지만, `B=1` 은 어떻게 될까?"""),
("md", """## 파트 3. MNIST: 같은 6만 장, 54% vs 96%

은닉층 하나(64개, ReLU)인 신경망을 1에폭만 학습시킨다. 바뀌는 건 배치 크기와 학습률뿐이다. 처음 실행하면 MNIST(약 11MB)를 내려받는다."""),
("code", """import time, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets

tr = datasets.MNIST("data", train=True, download=True); te = datasets.MNIST("data", train=False, download=True)
Xtr = tr.data.float().div(255).view(-1, 784); ytr = tr.targets
Xte = te.data.float().div(255).view(-1, 784); yte = te.targets
N = len(ytr)

def one_epoch(Xtr, ytr, Xte, yte, B, lr):
    N = len(ytr)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))
    start = time.time()
    for i in range(0, N, B):
        idx = perm[i:i + B]                              #@ANS idx = None              # TODO: same slicing as part 2
        opt.zero_grad()
        F.cross_entropy(model(Xtr[idx]), ytr[idx]).backward()
        opt.step()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    return acc, time.time() - start

for B, lr in [(N, 2.0), (32, 0.3)]:                     # each with its best learning rate (as in the slides)
    acc, sec = one_epoch(Xtr, ytr, Xte, yte, B, lr)
    print(f"B={B:6d} ({-(-N // B):5d} steps)  test accuracy {acc:.1%}  ({sec:.1f} s)")"""),
("code", """# Try other batch sizes (the learning rate may need to change too)
for B, lr in [(8, 0.1), (128, 1.0), (1024, 0.3)]:
    acc, sec = one_epoch(Xtr, ytr, Xte, yte, B, lr)
    print(f"B={B:6d} ({-(-N // B):5d} steps)  test accuracy {acc:.1%}  ({sec:.1f} s)")"""),
("md", """**확인 질문**
1. 파트 1 에서 `B` 가 1 이어도 1,000개의 평균이 전체 기울기와 거의 같은 이유는? (슬라이드의 기댓값 유도)
2. 파트 2 에서 `perm` 을 섞지 않고 `np.arange(m)` 으로 두면 무엇이 달라질까? 데이터가 정렬되어 있었다면?
3. 파트 3 에서 1에폭은 몇 걸음인가? `B=32` 와 `B=60000` 각각.
4. `B=1` 이 정답률은 좋은데 실전에서 잘 쓰지 않는 이유는? 걸린 시간을 비교해 보자.

데이터 출처: Efron et al. (2004) diabetes data (scikit-learn) · LeCun, Cortes, Burges, MNIST."""),
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
