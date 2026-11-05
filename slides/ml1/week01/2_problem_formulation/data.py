"""문제 정형화와 평가지표 차시 공유 숫자 (슬라이드 · 그림 · 실습이 같은 값을 쓴다). 덱 전체가 야구 데이터 하나로 간다.
exp_hr.py → assets/hr.json (1쪽 질문: 2024 MLB 타석 · 홈런, 오타니)
exp_baseball.py → assets/baseball.json (타구 두 문제 · 다음 시즌 홈런 수 · 타석 정확도의 함정)
assets/formulation.json 은 회귀 지표 손 계산 (옛 와인 · 항공 실험은 _이전/)"""
import os, json
_H = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(_H, "assets", "formulation.json")))

# ---- 1쪽 질문: 2024 MLB 홈런 (전부 '홈런 아님' 이라 답하는 모델)
HR = json.load(open(os.path.join(_H, "assets", "hr.json")))
assert (HR["mlb_pa"], HR["mlb_hr"], HR["ohtani_pa"], HR["ohtani_hr"]) == (182449, 5453, 731, 54)
assert (round(HR["mlb_acc_zero"] * 100, 2), round(HR["ohtani_acc_zero"] * 100, 2)) == (97.01, 92.61)

# ---- 야구 (assets/baseball.json, 덱 머리말의 \bb... 숫자와 같은지)
BB = json.load(open(os.path.join(_H, "assets", "baseball.json")))
BAT, NEXT, PA = BB["batted"], BB["next"], BB["pa"]
assert (BAT["n"], BAT["hr"], BAT["n_test"], BAT["hr_test"], BAT["model_caught"]) == (20506, 915, 6152, 275, 113)
assert (round(BAT["acc_base"] * 100, 1), round(BAT["acc_model"] * 100, 1), round(BAT["mae_base"], 1), round(BAT["mae_model"], 1)) == (95.5, 96.7, 37.5, 25.0)
assert (NEXT["n_train"], NEXT["n_test"]) == (394, 207)
assert [round(NEXT[k], 2) for k in ("mae_mean", "mae_last", "mae_knn", "mae_lin", "mae_leak")] == [7.35, 6.42, 6.65, 5.66, 5.01]
assert (PA["n"], PA["hr"], PA["sluggers"], PA["called"], PA["tp"], PA["fp"], PA["fn"], PA["players"], PA["players_hr"]) == \
       (182449, 5453, 29, 16792, 800, 15992, 4653, 651, 522)
assert (round(PA["acc_rule"] * 100, 2), round(PA["precision"] * 100, 2), round(PA["recall"] * 100, 1), round(PA["players_rate"] * 100, 1)) == (88.68, 4.76, 14.7, 80.2)

# ---- 회귀 지표 손 계산: 오차 [1, 1, 1, 9]
TOY = F["toy"]; assert TOY["mae"] == 3 and round(TOY["rmse"], 2) == 4.58

