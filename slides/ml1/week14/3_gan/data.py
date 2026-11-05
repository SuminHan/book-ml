"""슬라이드·그림·실습이 공유하는 숫자.
논문: Goodfellow 외 (2014) arXiv:1406.2661 'Generative Adversarial Nets' (판별자 최적일 때 D = 1/2).
  이후: DCGAN (Radford 외 2015), StyleGAN (Karras 외 2018~), 확산 모델 (2020~) 로 주도권 이동.
실험: exp_gan.py → assets/gan.npz · gan.json (고리, 봉우리 8개 × 시드 5, 손글씨 숫자 10,000걸음)."""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(HERE, "assets", "gan.npz")); J = json.load(open(os.path.join(HERE, "assets", "gan.json")))
RING_SNAP = Z["ring_snap"].tolist(); RING_D = {int(k): v for k, v in J["ring_d"].items()}
EIGHT_STEPS = J["eight_steps"]; EIGHT = {int(k): v["modes"] for k, v in J["eight"].items()}
DIG = {int(k): v for k, v in J["digits"].items()}; DIG_D = {int(k): v for k, v in J["digits_d"].items()}

if __name__ == "__main__":
    assert [round(v, 2) for v in RING_D[5000]] == [0.5, 0.5] and round(RING_D[300][0], 2) == 0.63
    assert EIGHT[0][3] == 8 and EIGHT[2][2] == 8 and EIGHT[3][:10] == [0, 0, 0, 0, 1, 1, 0, 1, 0, 3] and EIGHT[3][10] == 8
    assert sum(EIGHT[s][3] == 8 for s in range(5)) == 4                    # 1,500걸음에 8개 모두: 시드 5개 중 4개
    assert DIG[1000][0] == [0, 0, 0, 0, 0, 2000, 0, 0, 0, 0]               # 1,000걸음: 전부 5
    assert sum(c > 0 for c in DIG[3000][0]) == 5 and min(DIG[10000][0]) == 115 and round(DIG[10000][1] * 100) == 36
    print("data ok")
