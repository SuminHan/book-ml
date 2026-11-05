"""정규화와 표준화 차시 공유 숫자 (슬라이드 · 그림 · 실습이 같은 값을 쓴다).
와인: exp_scale.py → assets/scale.json (쇼츠 lessons/ml1/feature_scaling 과 같은 실험: UCI Wine 178병, MLP 13-16-3 tanh, 시드 10개 평균)
데이터 종류별: exp_modal.py → assets/modal.json (MNIST · CIFAR-10 · SMS · 항공 승객 · 숫자 말소리 · 머이브리지 말)"""
import os, json, numpy as np
_H = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(_H, "assets", "scale.json")))
M = json.load(open(os.path.join(_H, "assets", "modal.json")))

# ---- 와인 (쇼츠와 같은 숫자)
ACC = {k: round(v * 100) for k, v in J["acc"].items()}             # 그대로 72 · Min-Max 98 · Z-score 97
SAT = {k: round(v * 100) for k, v in J["sat"].items()}             # 학습 전 tanh 포화 98 · 0 · 0
CURVE = {k: np.array(v) for k, v in J["curve"].items()}            # 학습 손실 (10번마다)
NAMES = J["names"]; PRO, HUE = np.array(J["proline"]), np.array(J["hue"])
PRO_RNG, HUE_RNG = J["ranges"]["proline"], J["ranges"]["hue"]
BOUND = round(J["theory"], 2)                                        # 첫 층 가중치 ±1/√13
_N = J["neuron"]
SHOW = [("프롤린", "proline"), ("마그네슘", "magnesium"), ("알코올", "alcohol"), ("색조", "hue")]
NX = {k: _N["x"][NAMES.index(f)] for k, f in SHOW}; NZ = {k: _N["z"][NAMES.index(f)] for k, f in SHOW}
NW = {k: _N["w"][NAMES.index(f)] for k, f in SHOW}
RAW_SUM, Z_SUM = _N["raw"], _N["zs"]
MANY_RAW, MANY_Z = np.array(_N["many_raw"]), np.array(_N["many_z"])
PRO_MEAN, PRO_STD = float(PRO.mean()), float(PRO.std())
PRO_MM = (NX["프롤린"] - PRO_RNG[0]) / (PRO_RNG[1] - PRO_RNG[0])     # 1065 → 0.56
assert ACC == {"raw": 72, "minmax": 98, "z": 97} and SAT["raw"] == 98 and SAT["minmax"] == 0 and SAT["z"] == 0
assert BOUND == 0.28 and PRO_RNG == [278.0, 1680.0] and HUE_RNG == [0.48, 1.71]
assert (NX["프롤린"], round(NW["프롤린"], 2), round(NX["프롤린"] * NW["프롤린"]), round(RAW_SUM), round(np.tanh(RAW_SUM), 2)) == (1065.0, 0.15, 162, 168, 1.0)
assert (round(NZ["프롤린"], 2), round(Z_SUM, 2), round(float(np.tanh(Z_SUM)), 2)) == (1.01, 0.67, 0.58)
assert (round(PRO_MEAN), round(PRO_STD), round(PRO_MM, 2)) == (747, 314, 0.56)
assert min(MANY_RAW) > 40 and max(abs(np.tanh(MANY_Z))) < 0.8

# ---- 흑백 이미지: MNIST (학습 1만 장, MLP 784-64-10 tanh, SGD, 3에폭, 시드 3개 평균)
MN = M["mnist"]
MN_ACC = {k: round(v * 100) for k, v in MN["acc"].items()}
MN_SAT = {k: round(v * 100) for k, v in MN["sat"].items()}
MN_MEAN, MN_STD = round(MN["mean"], 4), round(MN["std"], 4)
DIGIT = np.array(MN["digit"], dtype=np.uint8)
assert MN_ACC == {"raw": 63, "div255": 90, "z": 92} and MN_SAT == {"raw": 95, "div255": 0, "z": 0}, (MN_ACC, MN_SAT)
assert (MN_MEAN, MN_STD, MN["lo"], MN["hi"]) == (0.1307, 0.3081, 0, 255)

# ---- 컬러 이미지: CIFAR-10 시험 1만 장, 채널별
CF = M["cifar"]; CF_MEAN = [round(v, 3) for v in CF["mean"]]; CF_STD = [round(v, 3) for v in CF["std"]]
CF_ONE = np.array(CF["one"], dtype=np.uint8)
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # torchvision 사전학습 모델이 쓰는 값
assert CF_MEAN == [0.494, 0.485, 0.45] and CF_STD == [0.247, 0.243, 0.262] and CF["shape"] == [10000, 32, 32, 3]

# ---- 언어: SMS 5,574통
SMS = M["sms"]
assert (SMS["n"], SMS["vocab_raw"], SMS["vocab_norm"]) == (5574, 15733, 8713)

# ---- 시계열: 항공 승객 (1949~1958 학습, 1959~1960 시험)
AIR = M["air"]; AIR_P = np.array(AIR["p"]); AIR_NTR = AIR["ntr"]
AIR_MM_TOP = (AIR["test_hi"] - AIR["lo"]) / (AIR["hi"] - AIR["lo"])
AIR_MAE = {"global": round(AIR["mae_global"]), "window": round(AIR["mae_window"])}
assert (AIR["lo"], AIR["hi"], AIR["test_hi"], round(AIR_MM_TOP, 2)) == (104, 505, 622, 1.29)
assert AIR_MAE == {"global": 74, "window": 10} and round(max(AIR["pred_global"])) == 487

# ---- 소리: 숫자 0 을 말한 6명 (8 kHz, int16)
SND = M["sound"]; PEAK = {k: round(v, 2) for k, v in SND["peak"].items()}
assert SND["sr"] == 8000 and SND["dtype"] == "int16" and PEAK["jackson"] == 0.74 and PEAK["theo"] == 0.02
PEAK_RATIO = round(SND["peak"]["jackson"] / SND["peak"]["theo"])
assert PEAK_RATIO == 37

# ---- 비디오: 머이브리지 말 GIF
VID = M["video"]
assert VID["shape"] == [15, 200, 300, 3] and VID["dtype"] == "uint8"
