"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: CIFAR-10 · CIFAR-100 (Hugging Face uoft-cs parquet, 처음 실행 때 data/ 에 내려받음, git 제외; 여기서는 lessons 실험 data 로 링크), 문장 임베딩 모델 all-MiniLM-L6-v2 (Hugging Face).
슬라이드 실험 exp_embed.py 는 15바퀴 (84.1%), 노트북은 5바퀴."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (embedding) 이 파일명 앞에 붙는다

CELLS = [
("md", """# 표상과 임베딩 실습

대상을 숫자 공간의 한 점으로. 닮은 것은 가깝게.

- **파트 1**: `nn.Embedding` = 원-핫 × 표, 코사인 유사도
- **파트 2**: 고양이 · 개만 배운 CNN 의 임베딩에 처음 보는 사진 넣기 (CIFAR)
- **파트 3**: 문장 임베딩으로 의미 검색

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요. 파트 2 는 데이터 내려받기 + 학습으로 몇 분 걸린다."""),
("code", """import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets"""),
("md", """## 파트 1. 표에서 한 줄 꺼내기"""),
("code", """W = torch.tensor([[0.2, 0.9], [0.8, 0.1], [0.7, 0.3], [0.1, 0.6]])
onehot = torch.tensor([0.0, 0.0, 1.0, 0.0])
print("one-hot @ W :", onehot @ W)

emb = nn.Embedding(4, 2)
emb.weight.data = W.clone()
print("nn.Embedding:", emb(torch.tensor(2)).detach())

def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))   #@ANS return None  # TODO: cosine similarity

print("cos(row 3, row 2) =", round(cosine(W[2].numpy(), W[1].numpy()), 2))
print("cos((1,0),(1,1)) =", round(cosine(np.array([1, 0]), np.array([1, 1])), 2), "  cos((1,0),(-1,0)) =", cosine(np.array([1, 0]), np.array([-1, 0])))"""),
("md", """## 파트 2. 고양이 · 개만 배운 신경망

CIFAR-10 의 고양이 (3) · 개 (5) 만으로 학습. 마지막 은닉층 64칸이 임베딩. 처음 보는 CIFAR-100 사진은 고양이 중심과 개 중심 중 어디에 가까울까?"""),
("code", """import os, io, urllib.request
import pandas as pd
from PIL import Image
HF = "https://huggingface.co/datasets/uoft-cs/"
FILES = {"c10_train": HF + "cifar10/resolve/main/plain_text/train-00000-of-00001.parquet",
         "c10_test": HF + "cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
         "c100_test": HF + "cifar100/resolve/main/cifar100/test-00000-of-00001.parquet"}
os.makedirs("data", exist_ok=True)
for name, url in FILES.items():
    if not os.path.exists(f"data/{name}.parquet"):
        urllib.request.urlretrieve(url, f"data/{name}.parquet")

def load(name, label_col):
    df = pd.read_parquet(f"data/{name}.parquet")
    images = np.stack([np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB")) for b in df["img"]])
    return images, df[label_col].values

def to_tensor(images):
    return torch.tensor(images).permute(0, 3, 1, 2).float() / 255 * 2 - 1

CAT, DOG = 3, 5
C100 = {"tiger": 88, "leopard": 42, "lion": 43, "orange": 53, "cloud": 23}     # CIFAR-100 class numbers
img_tr, ytr = load("c10_train", "label"); img_te, yte = load("c10_test", "label"); img100, y100 = load("c100_test", "fine_label")
keep = (ytr == CAT) | (ytr == DOG)
X_train, y_train = to_tensor(img_tr[keep]), torch.tensor((ytr[keep] == DOG).astype(int))
X_cat, X_dog = to_tensor(img_te[yte == CAT]), to_tensor(img_te[yte == DOG])
unseen = {name: to_tensor(img100[y100 == c]) for name, c in C100.items()}
print(len(X_train), "training images;", {k: len(v) for k, v in unseen.items()})"""),
("code", """class Net(nn.Module):
    def __init__(self):
        super().__init__()
        block = lambda a, b: [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU()]
        self.features = nn.Sequential(*block(3, 32), *block(32, 32), nn.MaxPool2d(2), *block(32, 64), *block(64, 64), nn.MaxPool2d(2),
                                      *block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc1, self.out = nn.Linear(128, 64), nn.Linear(64, 2)

    def embed(self, x):
        return F.relu(self.fc1(self.features(x)))       # the 64-number embedding

    def forward(self, x):
        return self.out(self.embed(x))

def train(X, y, epochs=5):
    torch.manual_seed(0)
    net = Net(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]; x = X[idx].clone()
            flip = torch.rand(len(x)) < 0.5; x[flip] = x[flip].flip(3)    # left-right flip
            loss = F.cross_entropy(net(x), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {ep}: loss {loss.item():.2f}")
    return net.eval()

net = train(X_train, y_train)
with torch.no_grad():
    acc = ((net(X_cat).argmax(1) == 0).float().sum() + (net(X_dog).argmax(1) == 1).float().sum()) / (len(X_cat) + len(X_dog))
print("test accuracy (cat vs dog):", round(acc.item(), 3))"""),
("code", """with torch.no_grad():
    E = {"cat": net.embed(X_cat).numpy(), "dog": net.embed(X_dog).numpy()}
    E.update({k: net.embed(v).numpy() for k, v in unseen.items()})
center_cat, center_dog = E["cat"].mean(0), E["dog"].mean(0)

def cat_side(X):
    d_cat = np.linalg.norm(X - center_cat, axis=1)
    d_dog = np.linalg.norm(X - center_dog, axis=1)
    return float((d_cat < d_dog).mean())                 #@ANS return None  # TODO: fraction closer to the cat center

for name, X in E.items():
    print(f"{name:8s} cat side: {cat_side(X):.0%}")"""),
("md", """## 파트 3. 문장 임베딩으로 의미 검색

문장 하나 → 384칸 벡터 (all-MiniLM-L6-v2, 영어). 단어가 겹치지 않아도 뜻이 가까우면 가깝다."""),
("code", """from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
lm = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()

def sentence_vectors(texts):
    t = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        h = lm(**t).last_hidden_state
    mask = t["attention_mask"][:, :, None]
    v = (h * mask).sum(1) / mask.sum(1)                  # average over the real tokens
    return F.normalize(v, dim=1).numpy()

docs = ["The cat is sleeping on the sofa.",
        "Stock prices fell sharply on Monday.",
        "A kitten naps in the living room.",
        "How to bake bread at home.",
        "The football team won the championship.",
        "Tigers are large wild cats."]
D = sentence_vectors(docs)
for query in ["my pet is taking a nap", "sports news", "big striped feline"]:
    scores = D @ sentence_vectors([query])[0]
    best = np.argsort(-scores)[:2]
    print(f"{query!r:28s} ->", [(docs[i], round(float(scores[i]), 2)) for i in best])"""),
("md", """**확인 질문**
1. 파트 1 에서 `nn.Embedding` 이 원-핫 곱셈보다 빠른 이유는?
2. 파트 2 에서 구름 · 오렌지가 고양이 쪽에 많이 놓였다면, 구름이 고양이를 닮았다는 뜻일까?
3. 파트 3 에서 "my pet is taking a nap" 은 단어가 하나도 겹치지 않는 문장을 찾았다. 어떻게 가능할까?
4. (도전) 파트 3 의 `docs` 에 여러분의 문장을 더하고, 검색이 틀리는 질문을 찾아보라.

데이터 출처: CIFAR-10 · CIFAR-100, A. Krizhevsky (2009). 문장 모델: Reimers & Gurevych (2019) Sentence-BERT, all-MiniLM-L6-v2."""),
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
