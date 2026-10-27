"""t-SNE 추가 실험 → assets/extra.npz
(1) 손글씨 숫자 perplexity 100 지도 (5 · 30 은 쇼츠 emb.npz 에 있음).
(2) 크기가 다른 두 무리: 10차원 정규분포 100점씩, 표준편차 1 (작은 무리) vs 10 (큰 무리), 중심 거리 50.
    원래 공간의 퍼짐 비 vs t-SNE 지도의 퍼짐 비 (perplexity 30). '섬 크기는 의미 없다' 를 보이기 위해.
(3) 같은 데이터에 실행할 때마다 (시드 0 · 1 · 2) 달라지는 지도."""
import os, numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
HERE = os.path.dirname(os.path.abspath(__file__))
X = load_digits().data
ts100 = TSNE(2, perplexity=100, random_state=0, init="pca").fit_transform(X)
rng = np.random.default_rng(0)
A = rng.normal(0, 1, (100, 10)); B = rng.normal(0, 10, (100, 10)); B[:, 0] += 50
C = np.r_[A, B]; lab = np.r_[np.zeros(100), np.ones(100)]
emb = TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(C)
spread = lambda Z: np.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean())
orig_ratio = spread(B) / spread(A); map_ratio = spread(emb[lab == 1]) / spread(emb[lab == 0])
print("orig spread ratio", round(orig_ratio, 1), "map ratio", round(map_ratio, 2))
seeds = [TSNE(2, perplexity=30, random_state=s, init="random").fit_transform(X) for s in (1, 2)]
np.savez_compressed(os.path.join(HERE, "assets", "extra.npz"), ts100=ts100, two=emb, two_lab=lab, orig_ratio=orig_ratio, map_ratio=map_ratio, seed1=seeds[0], seed2=seeds[1])
