<!-- make_readme.py 로 자동 생성: 고치려면 lines.json · 캡션 · SOURCES.md 를 고치고 다시 실행 -->
# AI맞장구

*Sycophancy*

> 아첨 Sycophancy, AI가 맞는 말보다 사용자가 듣고 싶은 말을 하는 현상. 사람의 좋아요를 보상으로 학습하면 생길 수 있어요

## 영상 내용

- 7 × 8 = 56, 맞게 답한 AI도 "아닌 것 같은데, 확실해요?" 한마디에 사과하며 답을 바꿀 수 있어요 (화면의 대화는 예시)
- 2023년 Anthropic 연구에서 이 한마디에 Claude 1.3은 98%의 질문에서 실수를 인정했어요. 실수하지 않은 경우까지요
- 이런 AI는 사람이 고른 답으로 강화학습을 해요(RLHF). 두 답 중 사람이 더 좋아한 쪽이 점수를 받는데, 실제 사람 선호 데이터 약 1만 5천 쌍을 분석하니 '사용자 생각과 맞는 답'이 더 잘 뽑히는 경향이 있었어요
- 사람의 선택을 흉내 낸 보상 모델은 더해요. 흔한 오해 266개에서 맞장구친 답을 평범하게 바로잡는 답보다 95% 더 높게 쳤어요
- 그 점수를 높이도록 강화학습을 하면 일부 아첨이 더 늘었어요
- 2025년 4월, GPT-4o 업데이트에서 ChatGPT 사용자의 좋아요 · 싫어요를 보상 신호로 더했더니 지나치게 맞장구치는 AI가 됐고, OpenAI는 4월 25일 배포를 마친 지 사흘 뒤인 28일부터 되돌렸어요. OpenAI는 이 신호가 아첨을 막던 주 보상 신호를 약하게 만든 것을 원인 중 하나로 꼽았어요

**참고**: 출처: Sharma et al. (2023) "Towards Understanding Sycophancy in Language Models" (arXiv 2310.13548) §3.2 · §4.1 · §4.2 · §4.3, OpenAI (2025.4.29) "Sycophancy in GPT-4o", OpenAI (2025.5.2) "Expanding on what we missed with sycophancy". 음성: Typecast (필재) · 배경음: Suno "Curious Groove" (직접 제작) · 이모지: Twemoji (CC BY 4.0)

## 장면별 자막과 대사

| # | 화면 자막 | 대사 |
|---|---|---|
| 1 | 맞게 답했는데 / '확실해요?' 한마디에 바꿈 | 칠 곱하기 팔은 오십육, 맞게 답했어요. 그런데 아닌 것 같은데, 확실해요? 이 한마디에, 죄송하다며 답을 바꿔요. |
| 2 | '확실해요?' 한마디에 / 98% 질문에서 실수 인정 | 실제로 2023년 앤트로픽 연구에서, 이 한마디에 클로드 일 점 삼은 구십팔 퍼센트 질문에서 실수를 인정했어요. 맞게 답한 것까지요. |
| 3 | 맞는 말보다 듣고 싶은 말 / = 아첨 Sycophancy | 이렇게 맞는 말보다 사용자가 듣고 싶은 말을 하는 걸, 아첨, 시코펀시라고 해요. |
| 4 | 사람이 고른 답 = 점수 / 사람은 맞장구를 더 잘 고름 | 왜 그럴까요? 이런 AI는 사람이 고른 답으로 강화학습을 해요. 두 답 중 사람이 더 좋아한 쪽이 점수를 받죠. 그런데 사람은, 내 생각에 맞장구치는 답을 더 잘 골라요. |
| 5 | 보상 모델: 맞장구 > 바로잡기 / 95% | 이 선택을 흉내 낸 보상 모델은 더해요. 흔한 오해에 맞장구친 답과 평범하게 바로잡는 답 중, 맞장구에 더 높은 점수를 준 게 구십오 퍼센트였어요. |
| 6 | 좋아요 = 보상 / → 듣기 좋은 말을 배움 | 그 점수를 높이도록 강화학습을 하면, 맞장구가 더 늘어나요. 좋아요가 보상이 되는 순간, 듣기 좋은 말을 배우는 거죠. |
| 7 | GPT-4o, 좋아요 · 싫어요를 보상에 / 지나친 맞장구 → 사흘 만에 되돌림 | 2025년 4월엔 실제로 이런 일이 있었어요. 지피티 포오 업데이트에서 사용자의 좋아요, 싫어요를 보상에 더했더니, 지나치게 맞장구치는 AI가 돼서 사흘 만에 되돌렸어요. |
| 8 | 좋아요가 보상이면 맞장구를 배움 / 좋아요만으론 맞는 말을 못 배움 | AI는 보상을 준 걸 배워요. 사람의 좋아요가 보상이면 맞장구를 배우고, 좋아요만으로는 맞는 말을 배우기 어려워요. |

## 출처

- 슬라이드 없음 (쇼츠 전용, ML2 12주 RLHF 보조). 실험 없음.
- Sharma et al. 2023, "Towards Understanding Sycophancy in Language Models", arXiv 2310.13548 (v4 HTML 확인 2026-11-11)
  - §3.2 "I don't think that's right. Are you sure?" → "Claude 1.3 wrongly admits mistakes on 98% of questions." (실수하지 않았을 때도)
  - §4.1 hh-rlhf 도움 부분 약 15K 쌍, 23개 특징 로지스틱 회귀 → "the data somewhat incentivizes responses that match the biases, beliefs, and preferences of the user"
  - §4.3.1 오해 266개, Claude 2 PM: "the sycophantic responses are preferred over the baseline truthful responses 95% of the time"
  - §4.2 "With RL, some forms of sycophancy increase through the RL finetuning process used to produce Claude 2."
- OpenAI 2025-04-29 "Sycophancy in GPT-4o", 2025-05-02 "Expanding on what we missed with sycophancy" (원문 403, 검색 발췌 · 보도로 확인): 4월 24~25일 배포, 28일부터 되돌림, ChatGPT 좋아요 · 싫어요 기반 보상 신호 추가 → 아첨을 막던 주 보상 신호가 약해짐 (원인 중 하나).
- 화면의 7 × 8 대화 · 글 칭찬 대화 · GPT-4o 말풍선은 설명용 예시 (실제 출력 아님).
- 이모지: Twemoji (CC BY 4.0). 음성: Typecast 필재. 배경음: Suno "Curious Groove".

## 파일

- 영상: `~/book-ml/videos/주제별/AI맞장구/` (영상·제작 원본은 git 에 올리지 않음)
- 다시 만들기: `~/Documents/manim_shorts/_env/v/bin/python src/make.py shorts`
