"""문제 정형화와 평가지표 차시 공유 숫자 (슬라이드 · 그림 · 실습이 같은 값을 쓴다).
exp_formulation.py → assets/formulation.json (와인 두 문제 · 항공 승객 기준선)
exp_aml.py → assets/aml.json (IBM AML HI-Small, Kaggle 로그인 필요)"""
import os, json
_H = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(_H, "assets", "formulation.json")))

# ---- 와인: 문제 A (분류) · 문제 B (회귀)
WINE = F["wine"]
WINE_ACC = {"model": round(WINE["acc_model"] * 100), "base": round(WINE["acc_base"] * 100)}
WINE_MAE = {"model": round(WINE["mae_model"], 2), "base": round(WINE["mae_base"], 2)}
assert WINE["counts"] == [59, 71, 48] and WINE_ACC == {"model": 100, "base": 39}, WINE_ACC
assert WINE_MAE == {"model": 0.45, "base": 0.66} and (WINE["alc_lo"], WINE["alc_hi"]) == (11.03, 14.83)

# ---- 항공 승객: 기준선 vs 모델 (시험 2년 24달)
AIR_MAE = {k.replace("mae_", ""): round(v) for k, v in F["air"].items()}
assert AIR_MAE == {"last": 44, "season": 48, "knn": 74, "mean": 206}, AIR_MAE

# ---- 넷플릭스 프라이즈 (Bennett & Lanning 2007 · 2009 결과)
NETFLIX = dict(users=480_000, movies=18_000, ratings=100_000_000, withheld=3_000_000, cinematch_quiz=0.9514, cinematch_test=0.9525,
               target_test=0.8572, winner_test=0.8567)   # target = 0.9525 × 0.9 (대회 규칙), 우승 2009-09

# ---- 회귀 지표 손 계산: 오차 [1, 1, 1, 9]
TOY = F["toy"]; assert TOY["mae"] == 3 and round(TOY["rmse"], 2) == 4.58

# ---- IBM AML (있으면)
_A = os.path.join(_H, "assets", "aml.json")
HAVE_AML = os.path.exists(_A)
AML = json.load(open(_A)) if HAVE_AML else None
if HAVE_AML:   # 덱 머리말의 \aml... 숫자와 같은지
    assert (AML["n"], AML["pos"], AML["tp"], AML["fp"], AML["fn"], AML["n_acct"], AML["pos_acct"]) == (5078345, 5177, 4483, 596314, 694, 496999, 3376)
    assert (round(AML["rate"] * 100, 2), round(AML["acc_zero"] * 100, 2), round(AML["acc_rule"] * 100, 2)) == (0.10, 99.90, 88.24)
    assert (round(AML["rule_precision"] * 100, 2), round(AML["rule_recall"] * 100, 1), round(AML["rate_acct"] * 100, 2), AML["rule_format"]) == (0.75, 86.6, 0.68, "ACH")
