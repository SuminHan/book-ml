<!-- make_readme.py 로 자동 생성: 고치려면 lines.json · 캡션 · SOURCES.md 를 고치고 다시 실행 -->
# Ablation 테스트

*부품을 하나씩 빼 보고 재기*

> Ablation Test(어블레이션 테스트)는 딥러닝 모델의 부품을 하나씩 빼 보고, 성능이 떨어진 만큼을 그 부품의 몫으로 보는 실험이에요.

## 영상 내용

- 가중치가 수천만 개인 딥러닝은 들여다봐서는 어느 부품이 무슨 일을 하는지 알 수 없어요
- 이름은 뇌 실험에서 왔어요: 1820년대 프랑스 생리학자 플루랑스(Flourens)가 비둘기의 소뇌를 떼어 내자 균형이 무너졌고, 1975년 앨런 뉴얼(Allen Newell)이 이 말을 AI 연구로 가져왔어요
- Transformer 헤드 수만 바꿔 다시 학습: 1개 24.9 · 8개 25.8 · 16개 25.8 · 32개 25.4 (번역 점수 BLEU) → 8개면 충분
- 다 배운 모델에서 헤드 끄기: BERT는 40%를 꺼도 거의 그대로(여분), 번역 모델의 한 층은 헤드 하나만 남기면 −13.6점
- SHAP은 입력을 빼서 예측의 이유를, Ablation은 부품을 빼서 설계의 이유를 따져요

**참고**: 수치: Vaswani 외 2017 "Attention Is All You Need" 표 3 (영어→독일어, 개발 세트) / Michel·Levy·Neubig 2019 "Are Sixteen Heads Really Better than One?"

**참고**: 사진: A. A. E. Disdéri, 1867년 이전 (CC0, Paris Musées · 위키미디어 공용)

**참고**: 막대그래프 세로축은 23.5부터 시작해요. "떨어진 만큼 = 몫" 막대는 예시 그림이에요.

## 장면별 자막과 대사

| # | 화면 자막 | 대사 |
|---|---|---|
| 1 | Transformer 어텐션 헤드 8개 / 1개로 줄이면? | 트랜스포머의 어텐션 헤드는 8개. 1개면, 성능이 얼마나 떨어질까요? |
| 2 | 가중치 6500만 개 / 들여다봐서는 모른다 | 가중치만 6500만 개. 들여다봐서는 몰라요. |
| 3 | Ablation Test: 부품 하나를 빼고 / 떨어진 만큼 = 그 부품의 몫 | 그래서 하나씩 빼 봐요. 떨어진 만큼이 그 부품의 몫. 이게 어블레이션 테스트. |
| 4 | Ablation = 떼어 냄 (뇌 실험에서 온 말) / 소뇌를 떼자 균형이 무너졌다 → 소뇌의 몫 | 이름은 뇌 실험에서 왔어요. 1820년대 플루랑스는 비둘기의 소뇌를 떼어 내, 균형이 무너지는 걸 봤죠. 이 말을 1975년 뉴얼이 AI로 가져왔어요. |
| 5 | 1개 24.9 · 8개 25.8 · 16개 25.8 · 32개 25.4 / 많을수록 좋은 게 아니다 | 헤드 1개로 다시 학습하면, 25.8에서 24.9로. 16개는 그대로, 32개는 오히려 떨어져요. 8개면 충분했던 거죠. |
| 6 | BERT: 헤드 40% 꺼도 거의 그대로 (여분) / 번역 모델 한 층: 1개만 남기면 −13.6 | 다 배운 모델에서 꺼 보면, 버트는 헤드 40퍼센트를 꺼도 그대로. 번역 모델 한 층은, 하나만 남기자 13.6점 하락. |
| 7 | SHAP: 입력을 빼고 → 예측의 이유 / Ablation: 부품을 빼고 → 설계의 이유 | 샵은 입력을 빼서 예측의 이유를, 어블레이션은 부품을 빼서 설계의 이유를 따져요. |
| 8 | 하나씩 빼 보고 · 떨어진 만큼이 몫 / 빼도 그대로면 여분 | 정리하면, 하나씩 빼 보고, 떨어진 만큼이 몫, 빼도 그대로면 여분. |

## 출처

- Vaswani et al. (2017), "Attention Is All You Need", NeurIPS — Table 3 (A): 헤드 수만 바꾸고 다시 학습
  (EN-DE dev newstest2013 BLEU): h=1 24.9 · h=4 25.5 · **h=8 25.8 (base)** · h=16 25.8 · h=32 25.4. base 파라미터 65×10⁶
  - 주의: 책 `kor/src/ml1/chapter12/3.md` 의 "EN-DE 24.16 → 23.48, EN-FR 38.19 → 37.19"는 원논문 Table 3과 다름 (확인 필요)
- Michel, Levy, Neubig (2019), "Are Sixteen Heads Really Better than One?", NeurIPS — 다 학습한 모델에서 헤드를 끄기만 함(재학습 X)
  - BERT-base(12층×12헤드=144) MNLI: 중요도 낮은 순으로 40%까지 꺼도 눈에 띄는 손해 없음 (§4.2, Fig. 3b)
  - WMT large Transformer(6층, 층당 16헤드, EN-FR, BLEU 36.05): 6층 인코더-디코더 어텐션을 헤드 1개만 남기면 −13.56 (Table 2)
- 쇼츠의 BLEU는 "번역 점수"로 부름. 막대그래프 세로축은 23.5부터 (화면에 표시)

### 이름의 유래 장면 (s3b)
- Flourens (1794–1867): 1822–24년 비둘기 등에서 뇌 부위를 떼어 내는(ablation) 실험, 소뇌 제거 → 균형·운동 협응 상실. 1824 *Recherches expérimentales sur les propriétés et les fonctions du système nerveux*
- 사진: Wikimedia Commons "Portrait de Pierre, Jean, Marie Flourens (1794-1867), (physiologiste), PH50650 (1 of 2).jpg", André-Adolphe-Eugène Disdéri, 1867년 이전, **CC0** (Paris Musées). 상반신만 잘라 rembg u2net_human_seg 누끼 → `flourens_cut.png`
- Allen Newell: 1974년 음성 인식 튜토리얼(1975 출판)에서 AI 시스템 분석에 ablation이라는 말을 씀 (Wikipedia "Ablation (artificial intelligence)"). 자유 라이선스 사진이 저해상도뿐이라 이름만 표기

## 파일

- 영상: `~/book-ml/videos/주제별/Ablation테스트/` (영상·제작 원본은 git 에 올리지 않음)
- 다시 만들기: `~/Documents/manim_shorts/_env/v/bin/python src/make.py shorts`
