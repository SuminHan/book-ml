"""문제 정형화와 평가지표 차시 공유 숫자 (슬라이드 · 그림 · 실습이 같은 값을 쓴다). 덱 전체가 야구 데이터 하나로 간다.
exp_baseball.py → assets/baseball.json (2024-06 모든 투구: 1쪽 질문 · 정확도의 함정 · 볼카운트, 던지기 전 두 문제)
assets/formulation.json 은 회귀 지표 손 계산 (옛 와인 · 항공 실험은 _이전/)"""
import os, json
_H = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(_H, "assets", "formulation.json")))

# ---- 야구 (assets/baseball.json, 덱 머리말의 \bb... 숫자와 같은지)
BB = json.load(open(os.path.join(_H, "assets", "baseball.json")))
PRE, PT = BB["pre"], BB["pitch"]
assert (PT["n"], PT["hr"], PT["tp"], PT["fp"], PT["fn"], PT["n_pa"], PT["n_batters"], PT["batters_hr"]) == (116355, 921, 126, 8063, 795, 30060, 480, 310)
assert [round(PT[k] * 100, 2) for k in ("rate", "acc_zero", "acc_rule", "precision")] == [0.79, 99.21, 92.39, 1.54]
assert (round(PT["recall"] * 100, 1), round(PT["pa_rate"] * 100, 2), round(PT["batters_rate"] * 100, 1), round(PT["pitches_per_pa"], 2)) == (13.7, 3.06, 64.6, 3.87)
assert (PT["ohtani_pitches"], PT["ohtani_hr"]) == (498, 12)

assert (PRE["n"], PRE["n_train"], PRE["n_test"]) == (115940, 88024, 27916)
assert [f"{PRE[k] * 100:.1f}" for k in ("swing_rate", "acc_base", "acc_count", "acc_model")] == ["48.2", "51.4", "59.6", "60.3"]
assert [f"{PRE[k]:.1f}" for k in ("mean_kmh", "mae_base", "mae_model", "mae_leak")] == ["143.6", "8.0", "7.3", "2.3"]

# ---- 회귀 지표 손 계산: 오차 [1, 1, 1, 9]
TOY = F["toy"]; assert TOY["mae"] == 3 and round(TOY["rmse"], 2) == 4.58


