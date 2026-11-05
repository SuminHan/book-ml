"""문제 정형화와 평가지표 차시 공유 숫자 (슬라이드 · 그림 · 실습이 같은 값을 쓴다). 덱 전체가 야구 데이터 하나로 간다.
exp_baseball.py → assets/baseball.json (2024-06 모든 투구: 1쪽 질문 · 정확도의 함정 · 볼카운트, 타구 두 문제, 다음 시즌 홈런 수)
assets/formulation.json 은 회귀 지표 손 계산 (옛 와인 · 항공 실험은 _이전/)"""
import os, json
_H = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(_H, "assets", "formulation.json")))

# ---- 야구 (assets/baseball.json, 덱 머리말의 \bb... 숫자와 같은지)
BB = json.load(open(os.path.join(_H, "assets", "baseball.json")))
BAT, NEXT, PT = BB["batted"], BB["next"], BB["pitch"]
assert (PT["n"], PT["hr"], PT["tp"], PT["fp"], PT["fn"], PT["n_pa"], PT["n_batters"], PT["batters_hr"]) == (116355, 921, 126, 8063, 795, 30060, 480, 310)
assert [round(PT[k] * 100, 2) for k in ("rate", "acc_zero", "acc_rule", "precision")] == [0.79, 99.21, 92.39, 1.54]
assert (round(PT["recall"] * 100, 1), round(PT["pa_rate"] * 100, 2), round(PT["batters_rate"] * 100, 1), round(PT["pitches_per_pa"], 2)) == (13.7, 3.06, 64.6, 3.87)
assert (PT["ohtani_pitches"], PT["ohtani_hr"]) == (498, 12)
assert (BAT["n"], BAT["hr"], BAT["n_test"], BAT["hr_test"], BAT["model_caught"]) == (20506, 915, 6152, 275, 115)
assert (round(BAT["acc_base"] * 100, 1), round(BAT["acc_model"] * 100, 1), round(BAT["mae_base"], 1), round(BAT["mae_model"], 1)) == (95.5, 96.8, 37.4, 24.8)
assert (NEXT["n_train"], NEXT["n_test"]) == (394, 207)
assert [round(NEXT[k], 2) for k in ("mae_mean", "mae_last", "mae_knn", "mae_lin", "mae_leak")] == [7.35, 6.42, 6.65, 5.66, 5.01]

# ---- 회귀 지표 손 계산: 오차 [1, 1, 1, 9]
TOY = F["toy"]; assert TOY["mae"] == 3 and round(TOY["rmse"], 2) == 4.58

