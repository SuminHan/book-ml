"""슬라이드·그림·실습이 공유하는 숫자.
실험 1 (exp_embed.py → assets/embed.npz): CIFAR-10 고양이 vs 개만 배운 작은 CNN (시험 84.1%), 마지막 은닉층 64칸 = 임베딩.
  처음 보는 CIFAR-100 사진 (호랑이 · 표범 · 사자 · 오렌지 · 사과 · 구름 각 100장) 이 고양이 중심과 개 중심 중 어디에 더 가까운가.
실험 2 (exp_clip.py → assets/clip.npz): CLIP ViT-B/32 (Radford 외 2021, 사진-글 4억 쌍) 로 같은 사진 20장씩, 종류별 평균 벡터의 코사인 유사도, 0-shot 분류.
쇼펜하우어 (1819) 『의지와 표상으로서의 세계』 첫 문장. 사진: J. Schäfer (1859), 퍼블릭 도메인.
손 계산: 단어 4개 · 2칸 임베딩 표에서 원-핫 (0, 0, 1, 0) 곱하기 = 3번째 줄."""
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(HERE, "assets", "embed.npz")); C = np.load(os.path.join(HERE, "assets", "clip.npz"))
ACC = float(Z["acc"])
NAMES = ["cat", "dog", "tiger", "leopard", "lion", "orange", "apple", "cloud"]
KO = dict(cat="고양이", dog="개", tiger="호랑이", leopard="표범", lion="사자", orange="오렌지", apple="사과", cloud="구름")
CC, CD = Z["cat"].mean(0), Z["dog"].mean(0)


def cat_side(X):
    return float((np.linalg.norm(X - CC, axis=1) < np.linalg.norm(X - CD, axis=1)).mean())


CAT_SIDE = {n: round(cat_side(Z[n]) * 100) for n in NAMES}
REF = np.r_[Z["cat"], Z["dog"]]; UNUSED = REF.std(0) < 0.01           # 고양이 · 개 사진에서는 거의 안 켜지는 칸
USES_UNUSED = {n: round(float((Z[n][:, UNUSED] > 0.05).any(1).mean()) * 100) for n in NAMES}
SIM = C["sim"]; ZS = dict(zip(NAMES, C["zs"].tolist()))
TABLE = np.array([[0.2, 0.9], [0.8, 0.1], [0.7, 0.3], [0.1, 0.6]])     # 손 계산 임베딩 표 (단어 4개 × 2칸)
ONEHOT = np.array([0, 0, 1, 0])

if __name__ == "__main__":
    assert round(ACC * 100, 1) == 84.1
    assert CAT_SIDE == {"cat": 88, "dog": 20, "tiger": 83, "leopard": 91, "lion": 50, "orange": 75, "apple": 77, "cloud": 99}, CAT_SIDE
    assert int(UNUSED.sum()) == 20 and USES_UNUSED["cat"] == 0 and USES_UNUSED["orange"] == 5 and USES_UNUSED["cloud"] == 29, USES_UNUSED
    r = lambda a, b: round(float(SIM[NAMES.index(a), NAMES.index(b)]), 2)
    assert r("tiger", "leopard") == 0.98 and r("cat", "dog") == 0.98 and r("orange", "apple") == 0.96 and r("cat", "orange") == 0.89
    assert ZS["apple"] == 1.0 and ZS["tiger"] == 0.4 and round(np.mean(list(ZS.values())) * 100, 1) == 78.1
    assert (ONEHOT @ TABLE).tolist() == TABLE[2].tolist()
    print("data ok")
