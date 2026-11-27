# 일반화 선형 모델 (GLM)

- 2026-11-27 새로 만듦. 앤드류 응 CS229 강의 노트 Part III (Generalized Linear Models) · 2018 4강을 덱 스타일로 옮김
  (노트 원본: `course-prep/ref/cs229_fall2021/cs229-notes1.pdf` 20쪽~).
- 자리: 계획 미정 (로지스틱회귀 다음 · 나이브 베이즈/SVM 앞이 앤드류 응 순서). 지금 폴더 번호 5 는 임시.
- 흐름: 기울기가 왜 같나 → y 의 종류 → 개수에 직선 (월드컵 골, 음수 · 약팀을 너무 낮게) → GLM 세 재료 → 지수족 → 베르누이 → 시그모이드가 나온다
  → 정규 → 선형회귀 · 포아송 → e^z (월드컵 Elo 100 마다 골 ×1.22) · 소프트맥스 (붓꽃) → 한 표 → 손실 = −log p → a'(η) = 평균 → (ŷ − y)x
  → 신경망 마지막 층 = GLM · 출력 함수와 손실 짝 (PyTorch) → 문제풀이 · 정리 · 참고 문헌
- 숫자: `data.py` (assert), 그림: `make_figs.py`. 데이터: 체스 (로지스틱 덱과 같은 파일), 월드컵 본선 1930~2026 1068경기 팀별 골 2136줄 (martj42/international_results, CC0, Elo 는 prep_worldcup.py 가 계산), 붓꽃 (sklearn).
- 아직 없음: 실습 노트북 (lab/).
