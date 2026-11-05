"""슬라이드·그림·실습이 공유하는 숫자.
논문: Sohl-Dickstein 외 (2015) arXiv:1503.03585 (처음 제안), Ho · Jain · Abbeel (2020) arXiv:2006.11239 DDPM (T = 1000, β 1e-4 → 0.02 선형),
  Song 외 (2020) DDIM (단계 건너뛰기), Rombach 외 (2021) arXiv:2112.10752 잠재 확산 = Stable Diffusion (512×512×3 → 64×64×4, 48분의 1).
실험: exp_ddpm.py → assets/ddpm.json · ddpm.npz (손글씨 숫자 작은 DDPM, 2,000장 생성, 숫자 분류기로 판정).
  VAE · GAN 비교 값은 같은 판정으로 잰 ../vae/assets/vae.json · ../gan/assets/gan.json.
그림: assets/sd/ (쇼츠 lessons/ml1/diffusion 의 sd_steps.py 로 Stable Diffusion v1.5 를 직접 돌린 결과, DDIM 50단계, 시드 7)."""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "assets", "ddpm.json"))); Z = np.load(os.path.join(HERE, "assets", "ddpm.npz"))
META = json.load(open(os.path.join(HERE, "assets", "sd", "meta.json")))
VAE_CONF = json.load(open(os.path.join(HERE, "..", "vae", "assets", "vae.json")))["vae"]["conf_rand"]
GAN_CONF = json.load(open(os.path.join(HERE, "..", "gan", "assets", "gan.json")))["digits"]["10000"][1]
ABAR = {int(k): v for k, v in J["abar"].items()}
T = 1000; COMPRESS = (512 * 512 * 3) // (64 * 64 * 4)

if __name__ == "__main__":
    assert [round(J[k]["conf"] * 100) for k in ("ddpm1000", "ddim50", "ddim10")] == [55, 49, 43]
    assert min(J["ddpm1000"]["counts"]) == 175 and max(J["ddpm1000"]["counts"]) == 225
    assert round(J["data_conf"] * 100) == 77 and round(VAE_CONF * 100) == 31 and round(GAN_CONF * 100) == 36
    assert round(ABAR[500], 3) == 0.079 and round(ABAR[1000], 5) == 0.00004 and round(ABAR[100], 2) == 0.90
    assert COMPRESS == 48 and META["steps"] == 50
    s, n = np.sqrt(ABAR[500]), np.sqrt(1 - ABAR[500]); assert round(s, 2) == 0.28 and round(n, 2) == 0.96
    print("data ok", {k: round(v, 5) for k, v in ABAR.items()})
