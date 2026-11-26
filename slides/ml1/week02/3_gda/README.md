# 가우시안 판별 분석 (GDA)

- 2026-11-27 새로 만듦. 앤드류 응 CS229 노트 Part IV (Generative Learning Algorithms) 의 GDA · 2018 5강을 덱 스타일로
  (노트 원본: `course-prep/ref/cs229_fall2021/cs229-notes2.pdf`). 옛 분포 덱 (`../7_distributions`, 보충) 의 "키로 남/여" 부분을 대신한다.
- 자리: 2주차 3차시 (GLM → 나이브 베이즈 → GDA → 다음 주 SVM). 나이브 베이즈가 베이즈 정리를 먼저 가르친다.
- 흐름: 판별 vs 생성 → 1차원 (꽃잎 길이, 공유 표준편차) → 베이즈 4.9 cm 49% → 사후확률 S 자 (로지스틱 58% 와 비교) → x² 가 지워지는 유도
  → 같은 꼴 다른 w → 다변량 가우시안 · Σ 읽기 → 2차원 GDA (직선, 94%) → 학습 = 세기 → QDA (곡선, 97%) → 생성 vs 판별 표 · 적은 데이터 실험
  → 가짜 꽃 만들기 → 나이브 베이즈와 같은 가족 → 문제풀이 · 정리 · 참고 문헌
- 숫자: `data.py` (assert), 그림: `make_figs.py`. 데이터: 붓꽃 versicolor · virginica 100송이 (sklearn).
- 아직 없음: 실습 노트북 (lab/).
