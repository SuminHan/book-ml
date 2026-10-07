<!-- make_readme.py 로 자동 생성: 고치려면 lines.json · 캡션 · SOURCES.md 를 고치고 다시 실행 -->
# BPR

*안 본 것 ≠ 싫은 것, 순서를 배우는 추천*

> Bayesian Personalized Ranking(BPR), 안 본 영화는 싫어하는 영화일까?

## 영상 내용

- 클릭·시청 기록에는 '봤다'만 남고 '싫다'는 없어요. 안 본 건 몰라서일 수도, 싫어서일 수도 있죠.

- • 그래서 2009년 Rendle 연구팀은 점수를 맞히는 대신 순서를 맞히자고 했어요. 본 영화가 안 본 영화보다 위에만 있으면 돼요.
- • (사용자, 그 사람이 본 영화, 안 본 영화) 세 개씩 무작위로 뽑아서, 본 영화 점수가 더 높아지게 조금씩 고쳐요(확률적 경사 하강법).
- • 수천 번 반복하면 순위표가 정리돼요. 안 본 영화끼리는 직접 비교한 적이 없지만, 그 순서는 비슷한 사람들의 기록에서 배워요(행렬 분해 모델).
- • 그래서 애니를 본 나에게는 인기 1위 스파이더맨(20명 중 16명이 봄)이 아니라 마리오가 추천 1위.
- • 같은 모델이라도 오디세이·스파이더맨을 본 민지에게는 옵세션이 추천 1위, 마리오는 꼴찌예요. 사람마다 다른 순위표 — 그래서 개인화(Personalized) 추천이에요.
- • 참고로 이름의 Bayesian은, 모델의 숫자가 지나치게 커지지 않을 거라는 사전 확률을 두고 푼 데서 왔어요(학습 식의 정규화 항).

- 참고: S. Rendle 외, "BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI 2009

**참고**: 시청 기록은 설명용 예시 데이터, 순위표는 실제로 BPR을 학습시킨 결과예요. 영화 포스터 출처: 각 영화 위키백과 문서

## 장면별 자막과 대사

| # | 화면 자막 | 대사 |
|---|---|---|
| 1 | 내가 안 본 영화 / 다 싫어하는 영화일까? | 내가 안 본 영화들, 다 싫어하는 영화일까요? |
| 2 | 기록엔 '봤다'만 있고 '싫다'는 없다 / 안 본 이유: 몰라서? 싫어서? | 기록엔 본 것만 남고, 싫어요는 없어요. 몰라서 안 봤을 수도, 싫어서일 수도 있죠. |
| 3 | BPR (Rendle 외, 2009) / 본 영화 > 안 본 영화, 순서만 맞히기 | 2009년 렌들 연구팀의 아이디어는, 점수 대신 순서. 본 영화가 안 본 영화보다 위에만 있으면 돼요. |
| 4 | (사용자, 본 영화, 안 본 영화) 셋씩 뽑아 / 본 영화 점수가 더 높아지게 | 사용자 한 명과, 그 사람이 본 영화, 안 본 영화를 뽑아서, 본 영화 점수가 더 높아지게 조금씩 고쳐요. |
| 5 | 수천 번 반복 → 순위표 정리 / 안 본 영화끼리 순서는 비슷한 사람들에게서 | 이걸 수천 번 반복하면 순위표가 정리돼요. 안 본 영화끼리의 순서는, 비슷한 사람들 기록에서 배우죠. |
| 6 | 애니를 본 나에게 추천 1위는 / 인기 1위 스파이더맨이 아니라 마리오 | 그래서 애니를 본 나에겐, 제일 인기 많은 스파이더맨 대신, 마리오가 추천 1위. |
| 7 | 사람마다 다른 순위표 / 나: 마리오 1위 · 민지: 옵세션 1위 | 같은 모델이라도, 오디세이와 스파이더맨을 본 민지에겐 순위가 달라요. 민지의 추천 1위는 옵세션, 마리오는 꼴찌죠. |
| 8 | 안 본 것 ≠ 싫은 것 / 본 영화 > 안 본 영화, 사람마다 다른 순서 | 정리하면, 안 본 건 싫은 게 아니고, 본 영화를 위로, 사람마다 다른 순서를 배워 추천해요. |

## 출처

파일 | 원본 | 비고
posters/*.jpg | 영어 위키백과 각 영화 문서의 인포박스 포스터 (비자유 저작물, 저해상도) | 사용자 판단으로 사용. 공개 업로드 시 저작권 유의
  odyssey   = The Odyssey (2026 film) poster.jpg
  spiderman = Spider-Man Brand New Day poster.jpg
  obsession = 사용자가 준 포스터(눈 클로즈업, 4:5)를 다른 포스터 비율로 가운데 크롭 · 이전 위키 포스터는 obsession_wiki.jpg
  toystory5 = Toy Story 5 poster.jpg
  doraemon  = New castle of the Undersea Devil.jpg (도라에몽: 신 진구의 해저귀암성)
  mario     = The Super Mario Galaxy Movie poster.jpeg (예비, 미사용)
사실 출처: B. Smith, G. Linden, J. York, "Amazon.com Recommendations: Item-to-Item Collaborative Filtering", IEEE Internet Computing (2003)
관람 기록(관객 20명)은 설명용 예시 데이터

사실 출처: S. Rendle, C. Freudenthaler, Z. Gantner, L. Schmidt-Thieme, "BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI 2009
mario.jpg 사용 (The Super Mario Galaxy Movie, 위키백과 포스터)

## 파일

- 영상: `~/book-ml/videos/주제별/BPR/` (영상·제작 원본은 git 에 올리지 않음)
- 다시 만들기: `~/Documents/manim_shorts/_env/v/bin/python src/make.py shorts`
