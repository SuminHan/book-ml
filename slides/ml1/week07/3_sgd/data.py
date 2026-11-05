"""SGD 차시 공유 데이터 (슬라이드 · 그림 · 실습 숫자 일치).
손 계산 예: 가장 단순한 모델 ŷ = b (점수 하나로 8명을 맞히기), 손실 ℓ_i = ½(b − y_i)², 기울기 ∂ℓ_i/∂b = b − y_i.
MNIST 숫자: assets/mnist_epoch.json (쇼츠 lessons/ml1/sgd 의 실험 그대로), assets/mnist_batch.json (exp_mnist.py)."""
import os, json, itertools, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))

# --- 손 계산 예: 8명의 점수, 처음 추측 b = 40, 학습률 α = 0.5
SCORES = [50, 56, 62, 66, 74, 78, 84, 90]
M = len(SCORES)
B0, ALPHA = 40, 0.5
MEAN = sum(SCORES) // M                                   # 70 = 최적의 b
GRADS = [B0 - y for y in SCORES]                          # 학생별 기울기 (b − y_i)
FULL_GRAD = sum(GRADS) // M                               # −30
GD_ONE = B0 - ALPHA * FULL_GRAD                           # 전체 배치 한 걸음 → 55
PAIR = (50, 74)                                           # 처음 뽑은 두 명
PAIR_GRAD = B0 - sum(PAIR) // 2                           # −22
PAIR_GRADS = [B0 - (a + b) / 2 for a, b in itertools.combinations(SCORES, 2)]   # 28가지 쌍
# 1에폭 = 미니배치 4개 (B = 2), 섞은 순서
EPOCH_BATCHES = [(50, 74), (84, 90), (56, 62), (66, 78)]
def sgd_epoch(b=B0):
    path = [b]
    for p in EPOCH_BATCHES:
        b = b - ALPHA * (b - sum(p) / 2); path.append(b)
    return path
SGD_PATH = sgd_epoch()                                    # 40 → 51 → 69 → 64 → 68

assert MEAN == 70 and FULL_GRAD == -30 and GD_ONE == 55 and PAIR_GRAD == -22
assert len(PAIR_GRADS) == 28 and np.mean(PAIR_GRADS) == FULL_GRAD and all(g == int(g) for g in PAIR_GRADS)
assert sorted(sum(EPOCH_BATCHES, ())) == SCORES and SGD_PATH == [40, 51, 69, 64, 68]

# --- MNIST (은닉층 1개 64, ReLU)
E = json.load(open(os.path.join(HERE, "assets", "mnist_epoch.json")))
Bj = json.load(open(os.path.join(HERE, "assets", "mnist_batch.json")))
N_MNIST, B_MNIST = 60000, 32
ACC0 = round(E["full"]["curve"][0][1] * 100)
ACC_FULL, ACC_SGD = round(E["full"]["acc"] * 100), round(E["b32"]["acc"] * 100)
FULL_STEPS = E["full_to_target"]["steps"]                 # 전체 배치로 96% 까지 걸음(= 에폭) 수
CURVE = [(s / 1000, a * 100) for s, a in E["b32"]["curve"]]
BATCH = Bj["batch"]                                       # [{B, steps, lr, acc, sec}]
COS = {int(k): v for k, v in Bj["cos"].items()}
IMAGENET = 1281167                                        # ImageNet-1k 학습 이미지 수 (ILSVRC 2012)
assert (ACC0, ACC_FULL, ACC_SGD, FULL_STEPS) == (13, 54, 96, 318) and N_MNIST // B_MNIST == 1875
assert [b["B"] for b in BATCH] == [1, 8, 32, 128, 1024, 60000] and min(COS[1]) < 0 < min(COS[32])
