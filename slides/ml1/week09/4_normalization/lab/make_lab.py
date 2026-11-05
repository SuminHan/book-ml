"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: scikit-learn load_digits (UCI Optical Recognition of Handwritten Digits)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (batchnorm) 이 파일명 앞에 붙는다

CELLS = [
("md", """# BatchNorm 실습

미니배치마다 뉴런별로 평균 0 · 표준편차 1, 그다음 γ · β.

- **파트 1**: BatchNorm 직접 구현 → 2, 4, 6, 8 재현, `nn.BatchNorm1d` 와 비교
- **파트 2**: 시그모이드 8층: 층별 퍼짐과 학습 곡선
- **파트 3**: `train()` / `eval()` 과 모아 둔 평균

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn"""),
("md", """## 파트 1. 직접 구현

$$\\hat x = \\frac{x - \\mu_B}{\\sqrt{\\sigma_B^2 + \\epsilon}}, \\qquad y = \\gamma \\hat x + \\beta$$"""),
("code", """def batchnorm(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=0)                                   # per neuron (column)
    var = x.var(axis=0)                                   # population variance (divide by m)
    x_hat = (x - mu) / np.sqrt(var + eps)                 #@ANS x_hat = None  # TODO: normalize
    return gamma * x_hat + beta

x = np.array([[2.0], [4.0], [6.0], [8.0]])               # one neuron, batch of 4
print("normalized:", batchnorm(x, 1.0, 0.0).ravel().round(2))
print("gamma=2, beta=1:", batchnorm(x, 2.0, 1.0).ravel().round(2))

bn = nn.BatchNorm1d(1); bn.train()
print("nn.BatchNorm1d:", bn(torch.tensor(x, dtype=torch.float32)).detach().numpy().ravel().round(2))"""),
("md", """## 파트 2. 시그모이드 8층

손글씨 숫자 1,797장. 먼저 학습 전 층별로 '그림 사이의 퍼짐' 을 재고, 그다음 SGD (lr 0.5, 배치 64) 로 1,500번 학습한다."""),
("code", """from sklearn.datasets import load_digits
d = load_digits()
X = torch.tensor(d.data / 16.0, dtype=torch.float32); y = torch.tensor(d.target)

def make_net(use_bn, depth=8, width=64):
    layers, i = [], 64
    for _ in range(depth):
        layers += [nn.Linear(i, width)] + ([nn.BatchNorm1d(width)] if use_bn else []) + [nn.Sigmoid()]   #@ANS layers += [nn.Linear(i, width)] + ([None] if use_bn else []) + [nn.Sigmoid()]  # TODO: the BatchNorm layer
        i = width
    return nn.Sequential(*layers, nn.Linear(width, 10))

def spreads(net, X):
    out, h = [], X
    for m in net:
        if isinstance(m, nn.Sigmoid):
            out.append(h.std(0).mean().item())           # spread across images of what enters the sigmoid
        h = m(h)
    return out

for use_bn in [False, True]:
    torch.manual_seed(0); net = make_net(use_bn); net.train()
    with torch.no_grad():
        s = spreads(net, X[:512])
    print("BN   " if use_bn else "plain", ["%.1e" % v for v in s])"""),
("code", """def train(net, X, y, steps=1500, lr=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(net.parameters(), lr); curve = []
    for it in range(steps + 1):
        if it % 50 == 0:
            net.eval()
            with torch.no_grad():
                curve.append((net(X).argmax(1) == y).float().mean().item())
            net.train()
        idx = torch.randint(0, len(X), (64,), generator=g)
        opt.zero_grad(); nn.functional.cross_entropy(net(X[idx]), y[idx]).backward(); opt.step()
    return curve

curves = {}
for use_bn in [False, True]:
    torch.manual_seed(0); curves[use_bn] = train(make_net(use_bn), X, y)
    print("BN" if use_bn else "plain", " accuracy at 250:", round(curves[use_bn][5], 3), " at 1500:", round(curves[use_bn][-1], 3))
steps = np.arange(0, 1501, 50)
plt.plot(steps, curves[False], label="plain"); plt.plot(steps, curves[True], label="BatchNorm")
plt.xlabel("step"); plt.ylabel("accuracy"); plt.legend(); plt.show()"""),
("md", """## 파트 3. 시험 모드: 모아 둔 평균"""),
("code", """torch.manual_seed(0); net = make_net(True); train(net, X, y, steps=500)
first_bn = net[1]
print("running mean (first 5 neurons):", first_bn.running_mean[:5].numpy().round(2))
one = X[:1]
net.eval()
with torch.no_grad():
    p_eval = net(one).argmax(1).item()                    #@ANS p_eval = None  # TODO: predict one image in eval mode
print("eval mode prediction:", p_eval, " true:", y[0].item())
net.train()
try:
    net(one)
except Exception as e:
    print("train mode with one image ->", type(e).__name__, ":", str(e)[:80])"""),
("md", """**확인 질문**
1. 파트 1 에서 직접 구현한 값과 `nn.BatchNorm1d` 의 값이 같은가? (`var` 를 m 으로 나누는지 m−1 로 나누는지 확인)
2. 파트 2 에서 BatchNorm 없는 신경망의 층별 퍼짐은 층마다 대략 몇 분의 1 로 줄어드나?
3. 파트 2 에서 활성 함수를 ReLU 로 바꾸면 BatchNorm 없이도 학습되나?
4. 파트 3 에서 train 모드로 그림 한 장을 넣으면 왜 오류가 나나?

데이터 출처: Alpaydin & Kaynak, Optical Recognition of Handwritten Digits, UCI (scikit-learn `load_digits`) · Ioffe & Szegedy (2015) arXiv:1502.03167."""),
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
