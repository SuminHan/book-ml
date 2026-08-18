# Chapter 9. LLM 포스트트레이닝: RLHF와 정렬 (LLM Post-training: RLHF & Alignment)

2022년, OpenAI는 InstructGPT 논문에서 놀라운 관찰을 보고했다 — 파라미터가
100배 더 적은 모델이라도, 사람이 원하는 방식으로 다듬어졌다면 사전학습만
거친 거대 모델보다 사람들이 훨씬 더 선호했다. Chapter 4에서 배운
사전학습(다음 토큰 예측)은 "그럴듯한 텍스트"를 만들어내지만, 그게 곧
"사람이 원하는 대답"이라는 보장은 없다 — 사전학습 데이터에는 훌륭한 글도,
엉뚱하거나 무례한 글도 섞여 있기 때문이다. 이번 장은 사전학습된 모델을
"사람이 실제로 원하는 방향"으로 다듬는 **포스트트레이닝**(post-training)
단계를 다룬다 — 그리고 그 핵심 도구가, Chapter 8에서 배운 PPO다.

## 9.1 SFT: 사람이 쓴 이상적인 답변으로 다듬기

가장 단순한 포스트트레이닝은 **지도 파인튜닝**(Supervised Fine-Tuning,
SFT)이다: 사람이 직접 작성한 "이상적인 (질문, 답변)" 쌍을 모아, 사전학습된
모델을 그 데이터에 대해 평범한 지도학습으로 한 번 더 학습시킨다 —
Chapter 4의 다음 토큰 예측 손실 그대로, 다만 데이터가 인터넷 텍스트가
아니라 사람이 직접 쓴 고품질 예시로 바뀐다. 문제는 확장성이다: "좋은
답변이 무엇인가"를 사람이 매번 처음부터 **작성**하는 것은 비용이 크고,
모델이 만들 수 있는 무수한 질문 유형을 다 커버하기 어렵다.

## 9.2 RLHF: 보상모델을 거쳐 PPO를 적용한다

**RLHF**(Reinforcement Learning from Human Feedback)는 다른 전략을
쓴다 — 사람이 매번 "정답"을 새로 쓰는 대신, 모델이 만든 **여러 답변 중
어느 쪽이 더 나은지 고르기만** 하면 된다(비교가 작성보다 훨씬 쉽고
빠르다). 이 선호 데이터로 **보상모델**(Reward Model)을 학습시킨 뒤,
Chapter 8의 PPO로 그 보상모델의 점수를 최대화하도록 언어모델 자체를
정책(policy)처럼 취급해 학습시킨다.

**1단계 — 보상모델 학습**: 같은 질문 \\(x\\)에 대해 모델이 만든 두 답변
\\(y_w\\)(사람이 더 선호한 쪽, winner)와 \\(y_l\\)(덜 선호한 쪽,
loser)이 있을 때, 보상모델 \\(r_\phi(x,y)\\)가 \\(y_w\\)에 더 높은
점수를 주도록 학습한다. 브래들리-테리 모델(Bradley-Terry model)은 이
선호 확률을 시그모이드로 모델링한다:

\\[P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\\]

이 확률을 최대화하는 손실은 **정확히 Chapter 2의 로지스틱회귀와 같은
교차 엔트로피 형태**다 — "정답 라벨"이 "\\(y_w\\)가 이겼다"는 사실
하나뿐인 이진분류 문제이기 때문이다:

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def reward_model_loss(r_win, r_lose):
    # Bradley-Terry: P(y_w > y_l) = sigmoid(r_win - r_lose)
    return -math.log(sigmoid(r_win - r_lose))
```

**2단계 — PPO로 정책 학습**: 보상모델이 준비되면, 언어모델 \\(\pi_\theta\\)를
"질문 \\(x\\)라는 상태에서, 답변 \\(y\\)라는 행동을 순차적으로(토큰 하나씩)
고르는 정책"으로 취급하고, 보상모델의 점수 \\(r_\phi(x,y)\\)를 보상 삼아
Chapter 8의 PPO를 그대로 적용한다. 다만 한 가지 항이 추가된다 — 사전학습된
원래 모델(SFT 모델) \\(\pi_{\text{ref}}\\)에서 너무 멀리 벗어나지
않도록, KL 발산 페널티를 목적함수에서 뺀다:

\\[J(\theta) = \mathbb{E}\left[r_\phi(x,y)\right] - \beta \, D_{KL}\big(\pi_\theta(\cdot|x) \,\|\, \pi_{\text{ref}}(\cdot|x)\big)\\]

왜 이 페널티가 필요한가: 보상모델도 결국 근사에 불과하므로, 정책이
보상모델의 허점을 악용해 점수만 높이고 실제로는 이상한 텍스트를 만들어낼
위험이 있다(**보상 해킹**, reward hacking). \\(\pi_{\text{ref}}\\)에서 너무
멀어지지 않도록 묶어두면, "말이 되는 문장" 안에서만 보상을 높이도록
제약할 수 있다 — Chapter 8.10에서 언급했던 "ChatGPT류 LLM을 RLHF로
조정한다"는 문장이 정확히 이 과정을 가리킨다.

## 9.3 DPO, PEFT/LoRA, 에이전트·RAG 개관

**DPO**(Direct Preference Optimization, 2023): RLHF는 (1) 보상모델을
따로 학습하고 (2) 그 보상모델로 PPO를 돌리는 2단계 과정이라 복잡하고
불안정할 수 있다. DPO는 수학적으로, RLHF 목적함수의 최적 정책이 보상함수를
\\(\pi_\theta\\)와 \\(\pi_{\text{ref}}\\)의 비율로 직접 표현할 수 있음을
보이고, 그 관계를 보상모델 손실에 대입해 **보상모델도 PPO도 없이, 선호
데이터에서 정책을 곧바로 학습**하는 손실을 유도한다:

\\[\mathcal{L}\_{\text{DPO}} = -\log \sigma\left(\beta \log
\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log
\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\\]

형태를 보면 여전히 9.2의 보상모델 손실과 똑같은 로지스틱 손실이다 —
다만 "보상"의 역할을 \\(\beta \log(\pi_\theta/\pi_{\text{ref}})\\)(정책이
기준 모델보다 그 답변을 얼마나 더 선호하게 됐는가)이 대신한다. 별도의
보상모델과 RL 루프 없이 지도학습 한 번으로 RLHF와 이론적으로 같은
지점에 도달한다는 것이 DPO의 매력이다.

**PEFT/LoRA**(Parameter-Efficient Fine-Tuning / Low-Rank Adaptation):
수십~수백억 개 파라미터 전체를 파인튜닝하는 것은 계산·저장 비용이 크다.
LoRA는 원래 가중치 \\(W\\)를 고정한 채, 훨씬 작은 두 저차원 행렬의 곱
\\(\Delta W = BA\\)(\\(B, A\\)의 차원이 \\(W\\)보다 훨씬 작다)만 학습해서
\\(W + \Delta W\\)를 새 가중치로 쓴다 — 학습해야 할 파라미터 수를
수백 분의 1로 줄이면서도 실전에서 전체 파인튜닝과 비슷한 성능을 낸다.

**에이전트와 RAG**(Retrieval-Augmented Generation): 프롬프팅만으로는
모델이 모르는 최신 정보나 외부 문서를 답할 수 없다. RAG는 질문과 관련된
문서를 먼저 검색해 프롬프트에 함께 넣어주고, 에이전트(agent)는 여기서
한 걸음 더 나아가 모델이 검색·계산기·코드 실행 같은 **도구를 스스로
호출**하도록 만든다 — 둘 다 "모델의 파라미터 안에 모든 지식을 우겨넣는"
대신 "모델이 필요할 때 바깥 정보/도구에 접근하게 한다"는 같은 방향의
해법이다.

**사전학습이 "무엇이든 그럴듯하게 잇는 법"을 가르친다면, 포스트트레이닝은
그중 "사람이 실제로 원하는 것"만 골라내는 단계다 — 그리고 그 핵심
도구(PPO)는 이미 이번 학기에 강화학습 문제를 풀기 위해 배운 바로 그
알고리즘이다.**

---

## 연습문제

**1. (코딩)** 위 `reward_model_loss`를 완성하고, DPO 손실
`dpo_loss`를 작성하라(핵심 줄은 빈칸으로 남겨져 있다고 가정):

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def reward_model_loss(r_win, r_lose):
    # ADD ADDITIONAL CODE HERE!!
    # -log(sigmoid(r_win - r_lose))

def dpo_loss(logp_win_theta, logp_win_ref, logp_lose_theta, logp_lose_ref, beta):
    # ADD ADDITIONAL CODE HERE!!
    # log_ratio_win = beta * (logp_win_theta - logp_win_ref)
    # log_ratio_lose = beta * (logp_lose_theta - logp_lose_ref)
    # -log(sigmoid(log_ratio_win - log_ratio_lose))

print(round(reward_model_loss(r_win=2.0, r_lose=-1.0), 3))  # 0.049 -- 잘 구분하면 손실이 작다
print(round(reward_model_loss(r_win=-1.0, r_lose=2.0), 3))  # 3.049 -- 거꾸로면 손실이 크다
```

**2. (개념 서술)** RLHF의 목적함수에 있는 KL 발산 페널티(\\(-\beta
D_{KL}(\pi_\theta\|\pi_{\text{ref}})\\))를 만약 완전히 없앤다면 어떤
문제가 생길 수 있는지, "보상 해킹"이라는 용어를 써서 두세 문장으로
설명하라.

**3. (손유도, Tier C — 폴백 준비 대상)** 브래들리-테리 손실
\\(\mathcal{L} = -\log\sigma(r_\phi(x,y_w) - r_\phi(x,y_l))\\)를
\\(r_\phi(x,y_w)\\)로 미분한 결과가, Chapter 2의 로지스틱회귀 손실을
미분했을 때와 같은 \\((\text{예측} - \text{정답})\\) 꼴의 그래디언트로
정리됨을 보여라.

**힌트**: \\(z = r_\phi(x,y_w) - r_\phi(x,y_l)\\)로 치환하면
\\(\mathcal{L} = -\log\sigma(z)\\)이고, 이는 "정답 라벨이 항상 1인"
로지스틱회귀 손실 \\(-\log h_w(x)\\)와 정확히 같은 형태다. Chapter
2.6에서 유도한 \\(\frac{d}{dz}(-\log\sigma(z)) = \sigma(z) - 1\\)을
그대로 재사용해서 \\(\frac{\partial \mathcal{L}}{\partial
r_\phi(x,y_w)}\\)를 구하라.

**정확성 확인**: 구한 그래디언트의 부호를 보고, 모델이 이미
\\(y_w\\)를 확신 있게 선호하고 있을 때(\\(\sigma(z) \to 1\\))
그래디언트가 왜 0에 가까워지는지(즉 더 이상 크게 갱신하지 않는지)
한 문장으로 설명하라.
