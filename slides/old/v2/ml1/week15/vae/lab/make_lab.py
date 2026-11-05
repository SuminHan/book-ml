"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: sklearn digits (설치에 포함). 슬라이드 실험 exp_vae.py 와 같은 설정 (400바퀴)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (vae) 이 파일명 앞에 붙는다

CELLS = [
("md", """# VAE 변분 오토인코더 실습

그림 → 구름 $(\\mu, \\sigma)$ → 뽑은 점 $z = \\mu + \\sigma\\epsilon$ → 그림. 손실 = 복원 + KL.

- **파트 1**: KL 손 계산 · 재매개변수화
- **파트 2**: 손글씨 숫자로 AE 와 VAE 학습 → 2칸 지도
- **파트 3**: N(0, 1) 에서 뽑아 되살리기 · 지도 격자

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression"""),
("md", """## 파트 1. KL · 뽑기

칸 하나의 $\\text{KL}(N(\\mu, \\sigma^2) \\| N(0, 1)) = \\frac12(\\mu^2 + \\sigma^2 - 1 - \\ln\\sigma^2)$."""),
("code", """def kl_to_standard(mu, sigma):
    return 0.5 * (mu ** 2 + sigma ** 2 - 1 - np.log(sigma ** 2))   #@ANS return None  # TODO: KL of N(mu, sigma^2) from N(0, 1)

for mu, sigma in [(1, 0.5), (0, 1), (0, 2), (2, 1), (0, 0.1)]:
    print(f"mu = {mu}, sigma = {sigma}: KL = {kl_to_standard(mu, sigma):.2f}")

# Reparameterization: the randomness is only in eps, so gradients reach mu and sigma
mu = torch.tensor(1.0, requires_grad=True); sigma = torch.tensor(0.5, requires_grad=True)
eps = torch.tensor(-0.4)
z = mu + sigma * eps                                     #@ANS z = None  # TODO: sample z from the cloud using eps
z.backward()
print("z =", round(z.item(), 2), "  dz/dmu =", mu.grad.item(), "  dz/dsigma =", round(sigma.grad.item(), 2))"""),
("md", """## 파트 2. AE vs VAE

같은 구조: 인코더 64 → 256 → 2 (AE) 또는 (μ 2, log σ² 2) (VAE), 디코더 2 → 256 → 64."""),
("code", """digits = load_digits()
X = torch.tensor(digits.data / 16.0, dtype=torch.float32); y = digits.target

def train(X, vae, epochs=400, seed=0):
    torch.manual_seed(seed)
    enc = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 4 if vae else 2))
    dec = nn.Sequential(nn.Linear(2, 256), nn.ReLU(), nn.Linear(256, 64), nn.Sigmoid())
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), 1e-3)
    for ep in range(epochs):
        for idx in torch.randperm(len(X)).split(128):
            x = X[idx]; h = enc(x)
            if vae:
                mu, logvar = h[:, :2], h[:, 2:]
                z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
                kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum()   # sum of the per-dimension KL
            else:
                z, kl = h, torch.tensor(0.0)
            rec = nn.functional.binary_cross_entropy(dec(z), x, reduction="sum")
            loss = (rec + kl) / len(x)                   #@ANS loss = None  # TODO: reconstruction + KL, per image
            opt.zero_grad(); loss.backward(); opt.step()
    return enc, dec

def encode(enc, X):
    with torch.no_grad():
        return enc(X)[:, :2].numpy()                     # the point (AE) or the cloud center mu (VAE)

models = {name: train(X, vae=(name == "VAE")) for name in ["AE", "VAE"]}
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
for ax, (name, (enc, dec)) in zip(axs, models.items()):
    M = encode(enc, X)
    ax.scatter(M[:, 0], M[:, 1], c=y, cmap="tab10", s=5)
    ax.add_patch(plt.Circle((0, 0), 2, fill=False, color="red", ls="--"))
    ax.set_title(f"{name}: mean {M.mean(0).round(1)}, std {M.std(0).round(2)}"); ax.set_aspect("equal")
plt.show()"""),
("md", """## 파트 3. N(0, 1) 에서 뽑아 되살리기"""),
("code", """clf = LogisticRegression(max_iter=2000).fit(digits.data / 16.0, y)   # judge which digit a picture looks like

torch.manual_seed(2)
z = torch.randn(2000, 2)
for name, (enc, dec) in models.items():
    with torch.no_grad():
        pics = dec(z).numpy()
    hist = np.bincount(clf.predict(pics), minlength=10) / len(pics)
    print(f"{name:3s} digits made from N(0,1):", (hist * 100).round(0), " kinds (>= 2%):", int((hist >= 0.02).sum()))"""),
("code", """def decode_grid(dec, n=9, lim=2.0):
    g = np.linspace(-lim, lim, n)
    z = torch.tensor([[a, b] for b in g[::-1] for a in g], dtype=torch.float32)
    with torch.no_grad():
        pics = dec(z).numpy().reshape(n, n, 8, 8)
    return pics.transpose(0, 2, 1, 3).reshape(n * 8, n * 8)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for ax, (name, (enc, dec)) in zip(axs, models.items()):
    ax.imshow(decode_grid(dec), cmap="gray_r"); ax.set_title(f"{name}: z in [-2, 2] x [-2, 2]"); ax.axis("off")
plt.show()"""),
("md", """**확인 질문**
1. 파트 1 에서 σ = 0.1 처럼 구름이 아주 작으면 KL 은 왜 커질까?
2. 파트 2 에서 AE 지도의 평균 · 퍼짐은 시드를 바꾸면 어떻게 될까? VAE 는?
3. 파트 3 에서 AE 의 격자 그림은 왜 몇 가지 숫자만 보일까?
4. (도전) VAE 손실의 `kl` 에 0.1 이나 10 을 곱해 다시 학습하면 지도와 그림은 어떻게 바뀌나?"""),
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
