"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: MNIST (torchvision 으로 내려받기). 슬라이드 실험 (exp_dropout.py) 은 100 에폭, 노트북은 시간 때문에 30 에폭."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (dropout) 이 파일명 앞에 붙는다

CELLS = [
("md", """# Dropout 실습

학습 때 뉴런을 무작위로 끄고, 시험 때는 모두 켠다.

- **파트 1**: 드롭아웃 직접 구현 (마스크 + 1/(1−p) 배)
- **파트 2**: MNIST 2,000장: 드롭아웃 없음 vs p = 0.5
- **파트 3**: `eval()` 을 깜빡하면?

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn"""),
("md", """## 파트 1. 드롭아웃 직접 구현

학습 때: 각 값을 확률 $p$ 로 0 으로, 살아남은 값은 $\\frac{1}{1-p}$ 배. 시험 때: 그대로."""),
("code", """def dropout(h, p, training, rng):
    if not training or p == 0:
        return h
    keep = rng.random(h.shape) >= p                       # True = this neuron stays on
    return h * keep / (1 - p)                             #@ANS return None  # TODO: zero the dropped ones, scale the rest

rng = np.random.default_rng(0)
h = np.array([2.0, 4.0, 6.0, 8.0])
print("one training pass:", dropout(h, 0.5, True, rng))
print("test:             ", dropout(h, 0.5, False, rng))
many = np.mean([dropout(h, 0.5, True, rng) for _ in range(10000)], axis=0)
print("average over 10,000 passes:", many.round(2), " (should be close to", h, ")")"""),
("md", """## 파트 2. 작은 데이터, 큰 신경망

MNIST 학습 2,000장 · 시험 1만 장. 784-512-512-10 신경망, 30 에폭 (몇 분 걸린다). 처음 실행하면 MNIST 를 내려받는다."""),
("code", """from torchvision import datasets
tr = datasets.MNIST("data", train=True, download=True); te = datasets.MNIST("data", train=False, download=True)
idx = torch.randperm(60000, generator=torch.Generator().manual_seed(0))[:2000]
Xtr = tr.data[idx].reshape(-1, 784).float() / 255; ytr = tr.targets[idx]
Xte = te.data.reshape(-1, 784).float() / 255; yte = te.targets

def make_net(p):
    return nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Dropout(p),
                         nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p), nn.Linear(512, 10))

def train(net, X, y, Xte, yte, epochs):
    opt = torch.optim.Adam(net.parameters(), 1e-3); hist = []
    for ep in range(epochs):
        net.train()                                       # dropout ON
        for b in torch.randperm(len(y), generator=torch.Generator().manual_seed(ep)).split(64):
            opt.zero_grad(); nn.functional.cross_entropy(net(X[b]), y[b]).backward(); opt.step()
        net.eval()                                        #@ANS pass  # TODO: switch dropout OFF before measuring
        with torch.no_grad():
            hist.append(((net(X).argmax(1) == y).float().mean().item(), (net(Xte).argmax(1) == yte).float().mean().item()))
    return np.array(hist)

results = {}
for p in [0.0, 0.5]:
    torch.manual_seed(0)
    results[p] = train(make_net(p), Xtr, ytr, Xte, yte, epochs=30)
    print(f"p={p}: train {results[p][-1, 0]:.3f}   test errors {round((1 - results[p][-1, 1]) * 10000)} / 10000")
for p, c in [(0.0, "r"), (0.5, "g")]:
    plt.plot(results[p][:, 0], c + "--", label=f"train p={p}"); plt.plot(results[p][:, 1], c, label=f"test p={p}")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.ylim(0.85, 1.005); plt.legend(); plt.show()"""),
("md", """## 파트 3. `eval()` 을 깜빡하면

같은 시험 그림 10장을 두 번 예측해 본다."""),
("code", """torch.manual_seed(0)
net = make_net(0.5)
_ = train(net, Xtr, ytr, Xte, yte, epochs=5)
first10 = Xte[:10]
net.train()                                               # forgot eval(): dropout still ON
with torch.no_grad():
    a, b = net(first10).argmax(1), net(first10).argmax(1)
print("train mode, run 1:", a.tolist()); print("train mode, run 2:", b.tolist())
net.eval()
with torch.no_grad():
    c, d = net(first10).argmax(1), net(first10).argmax(1)  #@ANS c, d = None, None  # TODO: predict twice in eval mode
print("eval mode,  run 1:", c.tolist()); print("eval mode,  run 2:", d.tolist()); print("true labels:      ", yte[:10].tolist())"""),
("md", """**확인 질문**
1. 파트 1 의 평균이 원래 값과 같아야 하는 이유는? 1/(1−p) 배를 안 하면 어떻게 되나?
2. 파트 2 에서 드롭아웃이 있을 때 학습 곡선이 더 울퉁불퉁한 이유는?
3. 파트 2 에서 p = 0.2, 0.8 로도 해 보자. 슬라이드의 결과 (663, 680) 와 비슷한 경향인가?
4. 파트 3 에서 train 모드의 두 예측이 다른 이유를 한 문장으로.

데이터 출처: LeCun, Cortes, Burges, MNIST (torchvision) · Hinton et al. (2012) arXiv:1207.0580."""),
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
