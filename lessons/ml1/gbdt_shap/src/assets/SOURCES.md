# 출처·수치
- 데이터: OpenML `titanic` (version 1) = Vanderbilt Biostatistics titanic3, 승객 1309명. 나이·요금 결측은 중앙값으로 채움(나이 그림에서는 결측 제외)
- 모델: sklearn GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0), 특징 = 등급·성별·나이·요금·가족 수(sibsp+parch)
- 정확도: StratifiedKFold(5, shuffle, rs=0) 평균 0.807 / 학습 오답률 1·10·200그루 = 38·20·14%
- SHAP 0.52 TreeExplainer(model_output="probability", 배경 100명) — 기준값 36.3%, 덧셈 오차 1e-8
- 예시 승객: 13번(1등석 여성 26살, 96%: +35 성별 +20 등급 +5 나머지), 609번(3등석 남성 26살, 13%: −9 −8 −6). 정수는 최대잔여 반올림(합 = 반올림 예측)
- 평균 |SHAP| (%p): 성별 19 · 등급 11 · 나이 7 · 요금 5 · 가족 2 / 10살 미만 나이 몫 평균 +30
- Shapley (1953), Lundberg & Lee (2017, NeurIPS)

## v2 (2026-10-02) — 섀플리 원리 장면 추가
- 사진: `titanic_stuart.jpg` = Wikimedia Commons "File:RMS Titanic 3.jpg", F. G. O. Stuart, 1912-04-10, Public domain. `titanic_crop.jpg` 는 잘라낸 것
- 영상의 몫은 `data.shapley()` 정확 계산: 팀원 3명(성별·등급·나머지=나이·요금·가족), 빠진 팀원은 전체 1309명 값으로 대입해 평균(v(∅)=38.2)
- 1등석 여성(13번): 성별→등급→나머지 순서 38→69→88→96 / 성별의 순서별 기여 31,31,37,39,28,39 → 평균 34.0 / 몫 38+34+23+1=96
- 3등석 남성(609번): 38 −15 −7 −3 = 13
- v1의 전체 중요도·나이 장면(5개 특징 TreeExplainer)은 60초 맞추느라 뺐음 (v1: shorts/_v1/, 영상 _v1.mp4)
- 얼굴 아이콘: Microsoft Fluent Emoji (MIT) woman_3d_light / man_3d_light → emoji_woman.png / emoji_man.png. 나레이션에서 "팀원" 대신 "승객 정보를 하나씩 알려 준다", GBDT 풀네임 읽기
- v3: 결정 트리 장면(실제 깊이 2 나무: 성별→남 나이<10 58/17%, 여 1·2등석 93/49%, 남성 64%) — 성별 모르면 0.64×17+0.36×93=44%. 3등석 남성 장면 축소
