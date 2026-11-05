"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: MNIST (torchvision). 슬라이드 실험 exp_resnet.py 와 같은 설정 (1만 장, 5 에폭)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (resnet) 이 파일명 앞에 붙는다

CELLS = [
("md", """# ResNet 잔차 신경망 실습

입력을 건너뛰어 더하는 지름길 $\\mathbf{y} = F(\\mathbf{x}) + \\mathbf{x}$.

- **파트 1**: 잔차 블록 직접 만들기
- **파트 2**: 손글씨 1만 장: 일반 vs 잔차, 6 · 18 · 34층 (몇 분 걸린다)
- **파트 3**: 층별 기울기 크기 + torchvision `resnet18` 구조

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torchvision import datasets"""),
("md", """## 파트 1. 잔차 블록"""),
("code", """class Block(nn.Module):
    def __init__(self, width, residual):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            return torch.relu(self.f(x) + x)              #@ANS return None  # TODO: F(x) + x, then ReLU
        return torch.relu(self.f(x))

def make_net(blocks, residual, width=64):
    return nn.Sequential(nn.Linear(784, width), nn.ReLU(), *[Block(width, residual) for _ in range(blocks)], nn.Linear(width, 10))

b = Block(4, residual=True)
nn.init.zeros_(b.f[2].weight); nn.init.zeros_(b.f[2].bias)  # make F(x) = 0
x = torch.tensor([[1.0, 2.0, 0.5, 3.0]])
print("input :", x.tolist()); print("output:", b(x).tolist(), " <- passes straight through when F = 0")"""),
("md", """## 파트 2. 깊이를 늘리면

블록 2 · 8 · 16개 = 층 6 · 18 · 34개. Adam (lr 1e-3), 배치 64, 5 에폭, 학습 정확도를 잰다."""),
("code", """tr = datasets.MNIST("data", train=True, download=True)
idx = torch.randperm(60000, generator=torch.Generator().manual_seed(0))[:10000]
X = tr.data[idx].reshape(-1, 784).float() / 255; y = tr.targets[idx]

def train(net, X, y, epochs=5):
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(epochs):
        for b in torch.randperm(len(y), generator=torch.Generator().manual_seed(ep)).split(64):
            opt.zero_grad(); nn.functional.cross_entropy(net(X[b]), y[b]).backward(); opt.step()
    with torch.no_grad():
        return (net(X).argmax(1) == y).float().mean().item()

results = {}
for blocks in [2, 8, 16]:
    for residual in [False, True]:
        torch.manual_seed(0)
        results[(blocks, residual)] = train(make_net(blocks, residual), X, y)   #@ANS results[(blocks, residual)] = None  # TODO: train and get training accuracy
        print(f"{2 * blocks + 2:2d} layers, residual={residual}: train accuracy {results[(blocks, residual)]:.3f}")"""),
("md", """## 파트 3. 층별 기울기와 실제 ResNet"""),
("code", """def first_layer_grad(net, X, y):
    net.zero_grad(); nn.functional.cross_entropy(net(X), y).backward()
    return net[0].weight.grad.abs().mean().item()

for residual in [False, True]:
    torch.manual_seed(0)
    print(f"34 layers, residual={residual}: |grad| of the first layer = {first_layer_grad(make_net(16, residual), X[:512], y[:512]):.2e}")

from torchvision.models import resnet18
r = resnet18(weights=None)
print(r.layer2[0])                                        # first block of stage 2: note the 'downsample' shortcut (1x1 conv)
print("total parameters:", f"{sum(p.numel() for p in r.parameters()):,}")"""),
("md", """**확인 질문**
1. 파트 1 에서 F 가 0 이면 출력이 입력과 같다. 일반 블록이었다면 같은 입력에 무엇이 나왔을까?
2. 파트 2 에서 일반 신경망 34층의 학습 정확도는? 과대적합인가?
3. 파트 3 에서 첫 층 기울기 크기는 몇 배 차이 나나? 지난 시간의 기울기 소실과 어떻게 연결되나?
4. `resnet18` 의 `downsample` 은 왜 있을까? (힌트: 채널 64 → 128, 크기 절반)

데이터 출처: LeCun, Cortes, Burges, MNIST (torchvision) · He et al. (2015) arXiv:1512.03385."""),
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
