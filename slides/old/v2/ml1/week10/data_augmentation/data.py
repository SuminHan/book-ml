"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/data_augmentation 과 같은 실험, exp_aug.py → assets/aug.json).
MNIST 숫자마다 10장 (100장) 만 학습, 작은 CNN 400 에폭, 시험 1만 장, 학습 세트 5개 평균. 증강 = 매 에폭 무작위 ±10도 회전 · ±2칸 이동.
AlexNet: Krizhevsky, Sutskever, Hinton (2012) — 256×256 에서 224×224 조각 + 좌우 뒤집기 → 학습 세트 2048배.
고양이 사진: AI 생성 이미지 (Stable Diffusion v1.5), 증강 9장은 쇼츠의 make_imgs.py 로 만든 것을 복사."""
import os, json
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "aug.json")))
N_TRAIN, BASE, AUG = J["n_train"], round(J["base"] * 100), round(J["aug"] * 100)
BASE_ALL = [round(v * 100, 1) for v in J["base_all"]]; AUG_ALL = [round(v * 100, 1) for v in J["aug_all"]]
ALEX_X = (256 - 224) * (256 - 224) * 2
CATS = ["flip", "rotl", "rot", "crop", "face", "bright", "dark", "gray", "color"]

if __name__ == "__main__":
    assert (N_TRAIN, BASE, AUG) == (100, 78, 87) and all(a > b for a, b in zip(AUG_ALL, BASE_ALL)) and ALEX_X == 2048
    print("data ok", BASE_ALL, AUG_ALL)
