"""슬라이드·그림·실습이 공유하는 숫자.
논문: He, Zhang, Ren, Sun (2015) arXiv:1512.03385, Table 2 (ImageNet 검증 top-1 오답률 %, 10-crop) · 152층 앙상블 top-5 3.57% (ILSVRC 2015 1위).
우리 실험: exp_resnet.py → assets/resnet.json (MNIST 1만 장, 블록 2 · 8 · 16, 5 에폭)."""
import os, json
ERR = {("plain", 18): 27.94, ("plain", 34): 28.54, ("res", 18): 27.88, ("res", 34): 25.03}
DEPTH_MAX, TOP5, YEAR = 152, 3.57, 2015
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "resnet.json")))
BLOCKS = (2, 8, 16)
LAYERS = {b: 2 * b + 2 for b in BLOCKS}                          # Linear 층 수 (블록마다 2 + 입력 · 출력)
TRAIN_END = {(b, r): round(J[f"{b}_{int(r)}"][-1][0] * 100, 1) for b in BLOCKS for r in (False, True)}
TEST_END = {(b, r): round(J[f"{b}_{int(r)}"][-1][1] * 100, 1) for b in BLOCKS for r in (False, True)}

if __name__ == "__main__":
    assert ERR[("plain", 34)] > ERR[("plain", 18)] and ERR[("res", 34)] < ERR[("res", 18)]
    assert TRAIN_END == {(2, False): 93.7, (2, True): 95.8, (8, False): 57.4, (8, True): 95.1, (16, False): 11.7, (16, True): 96.2}, TRAIN_END
    assert LAYERS == {2: 6, 8: 18, 16: 34}
    print("data ok", TEST_END)
