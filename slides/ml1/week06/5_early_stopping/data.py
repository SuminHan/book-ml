"""슬라이드·그림·실습이 공유하는 숫자 — 쇼츠 lessons/ml1/early_stopping 과 같은 실험 (실제 데이터 + 실제 계산: sklearn 손글씨 숫자 load_digits, 8×8 그림 1797장)
학습 500장 · 검증 500장(seed 0 무작위). 실험 조건: 학습 500장 중 20%(100장)의 라벨을 일부러 다른 숫자로 바꿔 둠(라벨 잡음 → 과대적합이 뚜렷해짐). 검증 라벨은 그대로. 영상에는 라벨 잡음을 보여 주지 않고 캡션·SOURCES 에만 밝힘
모델 = 작은 신경망(MLPClassifier, 은닉 256, adam lr 0.001), partial_fit 로 한 번(epoch)씩 300번 학습하며 매번 정확도를 잰다
Early stopping: 검증 정확도 최고 이후 PATIENCE 번 더 나아지지 않으면 멈추고 최고 시점 모델로 되돌림
assets/digits_run2.npz = 300번 학습 결과 캐시 (지우면 다시 계산, 1~2분)"""
import os, warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "assets", "digits_run2.npz")
NTR, NVA, NOISE, EPOCHS, PATIENCE = 500, 500, 0.2, 300, 10

IMG, Y = load_digits(return_X_y=True)
_r = np.random.default_rng(0); _idx = _r.permutation(len(Y))
TR, VA = _idx[:NTR], _idx[NTR:NTR + NVA]
FLIP = _r.choice(NTR, int(NOISE * NTR), replace=False)                       # 정답표가 틀린 학습 그림(TR 안의 위치)
Y_SHEET = Y[TR].copy()
Y_SHEET[FLIP] = (Y_SHEET[FLIP] + _r.integers(1, 10, len(FLIP))) % 10          # 다른 숫자로 바꿈

def _run():
    warnings.filterwarnings("ignore")
    m = MLPClassifier(hidden_layer_sizes=(256,), learning_rate_init=1e-3, random_state=0)
    acc_tr, acc_va, pred_tr, pred_va = [], [], [], []
    for _ in range(EPOCHS):
        m.partial_fit(IMG[TR] / 16, Y_SHEET, classes=np.arange(10))
        p = m.predict(IMG[TR] / 16); pred_tr.append(p)
        q = m.predict(IMG[VA] / 16); pred_va.append(q)
        acc_tr.append((p == Y_SHEET).mean()); acc_va.append((q == Y[VA]).mean())
    return np.array(acc_tr), np.array(acc_va), np.array(pred_tr), np.array(pred_va)

if os.path.exists(CACHE):
    _c = np.load(CACHE); ACC_TR, ACC_VA, PRED_TR, PRED_VA = _c["tr"], _c["va"], _c["pred"], _c["pred_va"]
else:
    ACC_TR, ACC_VA, PRED_TR, PRED_VA = _run(); np.savez_compressed(CACHE, tr=ACC_TR, va=ACC_VA, pred=PRED_TR, pred_va=PRED_VA)

BEST = int(np.argmax(ACC_VA)) + 1                                            # 검증 정확도 최고 = 몇 번째 학습
def stop_at(p):
    best, bi = -1, 0
    for i, v in enumerate(ACC_VA):
        if v > best: best, bi = v, i
        elif i - bi >= p: return i + 1
    return EPOCHS
STOP = stop_at(PATIENCE)
MEM = (PRED_TR[:, FLIP] == Y_SHEET[FLIP]).mean(1)                            # 틀린 정답표를 그대로 따라 답한 비율(=외움)
pct = lambda a: int(round(float(a) * 100))

assert len(FLIP) == 100
assert BEST == 31 and pct(ACC_VA[BEST - 1]) == 92
assert STOP == 41 == BEST + PATIENCE
assert pct(ACC_TR[-1]) == 97 and pct(ACC_VA[-1]) == 83                        # 300번: 학습 97%, 검증 83%
assert pct(ACC_TR[BEST - 1]) == 77                                            # 31번: 학습 77% (틀린 20%는 안 따라함)
assert pct(MEM[BEST - 1]) == 2 and pct(MEM[-1]) == 86                         # 틀린 정답 100장 중 외운 수: 2장 → 86장

SHOW_VA = [325, 115, 212, 483]                                                # 화면의 검증 그림(VA 안의 위치): 실제 1 · 0 · 6 · 4
LOST = int(((PRED_VA[BEST - 1] == Y[VA]) & (PRED_VA[-1] != Y[VA])).sum())    # 31번째엔 맞혔는데 300번째엔 틀린 검증 그림 수
assert [int(Y[VA[j]]) for j in SHOW_VA] == [1, 0, 6, 4]
assert all(PRED_VA[BEST - 1, j] == Y[VA[j]] for j in SHOW_VA)                # 31번째: 모두 맞힘
assert [int(PRED_VA[-1, j]) for j in SHOW_VA] == [2, 8, 5, 1]                 # 300번째: 1→2, 0→8, 6→5, 4→1
assert LOST == 59

PAT = {1: (21, 20, 90), 3: (23, 20, 90), 5: (36, 31, 92), 10: (41, 31, 92), 20: (51, 31, 92), 50: (81, 31, 92)}   # 참을성: (멈춘 번째, 되돌린 번째, 검증 %)
for _p, _v in PAT.items():
    _s = stop_at(_p); _b = int(np.argmax(ACC_VA[:_s])); assert (_s, _b + 1, pct(ACC_VA[_b])) == _v, _p

if __name__ == "__main__":
    print(BEST, STOP, pct(ACC_TR[0]), pct(ACC_VA[0]), pct(ACC_TR[BEST - 1]), pct(ACC_VA[BEST - 1]), pct(ACC_TR[-1]), pct(ACC_VA[-1]), pct(MEM[BEST - 1]), pct(MEM[-1]))
