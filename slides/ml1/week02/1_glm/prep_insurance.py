"""자동차 보험 데이터 준비: freMTPL2freq (프랑스 자동차 책임보험 678,013 계약, OpenML 41214) 에서
1년을 꼬박 채운 계약 (Exposure >= 1) 만 골라 data/insurance_claims.csv (bonus_malus, claims) 로 저장한다.
실행: ~/Documents/manim_shorts/_env/v/bin/python prep_insurance.py"""
import os
from sklearn.datasets import fetch_openml

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    d = fetch_openml(data_id=41214, as_frame=True, parser="auto").frame
    f = d[d.Exposure >= 1.0]
    out = f[["BonusMalus", "ClaimNb"]].astype(int).rename(columns={"BonusMalus": "bonus_malus", "ClaimNb": "claims"})
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    out.to_csv(os.path.join(HERE, "data", "insurance_claims.csv"), index=False)
    print(len(out), out.claims.mean(), out.bonus_malus.min(), out.bonus_malus.max())
