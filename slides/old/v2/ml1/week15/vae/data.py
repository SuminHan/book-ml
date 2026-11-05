"""슬라이드·그림·실습이 공유하는 숫자.
논문: Kingma & Welling (2013) arXiv:1312.6114 'Auto-Encoding Variational Bayes' — 재매개변수화 z = μ + σ ε.
실험: exp_vae.py → assets/vae.json · vae.npz (sklearn digits 1,797장, 잠재 2칸, AE vs VAE 같은 구조 · 400바퀴).
  숫자 종류 판정 = 원본으로 학습한 로지스틱 회귀. '0 → 1 걷기' 그림은 쇼츠의 assets/vae_shorts.npz (같은 설정의 VAE).
손 계산: 1칸 잠재, μ = 1, σ = 0.5, ε = −0.4."""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "assets", "vae.json"))); Z = np.load(os.path.join(HERE, "assets", "vae.npz"))
SH = np.load(os.path.join(HERE, "assets", "vae_shorts.npz"))
Y, X = Z["y"], Z["x"]
AE, VAE = S["ae"], S["vae"]
MU, SIG, EPS = 1.0, 0.5, -0.4
Z_HAND = MU + SIG * EPS
KL_HAND = 0.5 * (MU ** 2 + SIG ** 2 - 1 - np.log(SIG ** 2))

if __name__ == "__main__":
    assert [round(v, 1) for v in AE["lat_mean"]] == [1.1, 3.5] and [round(v, 1) for v in AE["lat_std"]] == [3.7, 3.4]
    assert [round(v, 1) for v in VAE["lat_mean"]] == [-0.0, 0.0] and [round(v, 2) for v in VAE["lat_std"]] == [0.98, 0.98]
    assert AE["n01_kinds"] == 5 and VAE["n01_kinds"] == 10
    assert round(max(AE["n01_hist"]) * 100) == 33 and round(min(VAE["n01_hist"]) * 100) == 2 and round(max(VAE["n01_hist"]) * 100) == 17
    assert round(AE["err"], 2) == 1.85 and round(VAE["err"], 2) == 2.10
    assert round(Z_HAND, 2) == 0.8 and round(KL_HAND, 2) == 0.82
    print("data ok")
