# Chapter 12. 모방학습과 인간 피드백 (Imitation Learning & Learning from Human Feedback)

Chapter 2~11에서 배운 모든 알고리즘은 에이전트가 스스로 시행착오를 겪으며
보상 신호로부터 배웠다. 그런데 어떤 문제는 "시행착오" 자체가 위험하거나
비싸다 — 로봇 팔이 사람 옆에서 물건을 옮기는 법을 무작위로 시도하며
배우게 둘 수는 없다. 이번 장은 보상을 직접 최적화하는 대신, **사람의
시연**이나 **사람의 선호**로부터 배우는 두 가지 방법을 다룬다.

## 12.1 모방학습: 보상 없이, 시연만으로

**모방학습**(Imitation Learning)의 가장 단순한 형태는 **행동
복제**(Behavior Cloning, BC)다 — 전문가(사람 또는 기존 정책)가 남긴
(상태, 행동) 시연 데이터를 모아서, "이 상태에서는 이 행동"이라는
지도학습 문제로 그대로 바꿔버린다. 보상함수를 설계할 필요도, 환경과
상호작용하며 탐험할 필요도 없다 — 로지스틱회귀나 다중분류와 정확히
같은 방식으로, 상태를 입력받아 행동(이산이면 분류, 연속이면 회귀)을
출력하도록 신경망을 학습시키면 끝이다.

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-max(-20, min(20, z))))

def train_behavior_cloning(demos, epochs, lr):
    # demos: [(state, action), ...] 전문가 시연, action은 0 또는 1(이진 예시)
    w, b = 0.0, 0.0
    for _ in range(epochs):
        for s, a in demos:
            pred = sigmoid(w * s + b)
            grad = pred - a          # 로지스틱회귀와 똑같은 그래디언트 형태
            w -= lr * grad * s
            b -= lr * grad
    return w, b
```

## 12.2 왜 행동 복제만으로는 부족한가: 복합 오차

행동 복제는 단순하지만 근본적인 약점이 있다 — 학습은 **전문가가 실제로
방문한 상태**들에서만 이뤄지는데, 배포된 정책은 작은 실수 하나로 전문가가
한 번도 가본 적 없는 상태로 빗나갈 수 있다. 그 낯선 상태에서는 정책이
무엇을 해야 할지 전혀 배운 적이 없으므로 더 큰 실수를 하고, 그 실수가
또 다른 낯선 상태로 이어진다 — 오차가 시간이 지날수록 눈덩이처럼
불어나는 **복합 오차**(compounding error) 문제다.

**DAgger**(Dataset Aggregation)는 이 문제를 반복으로 완화한다: (1)
지금까지 학습된 정책으로 직접 환경을 돌아다녀 보고, (2) 그 경로에서
마주친 상태들에 대해 전문가에게 "여기선 어떻게 해야 하나요?"라고 다시
물어서 정답을 받고, (3) 이 새 데이터를 원래 시연 데이터에 합쳐서 다시
학습한다. 이 과정을 반복하면, 학습 데이터가 점점 "정책이 실제로 방문하는
상태"들을 포함하게 되어 복합 오차가 줄어든다 — Chapter 6의 SARSA가
"실제로 따르는 정책"의 가치를 배우는 것과 비슷한 정신이다: **이론상
이상적인 상황이 아니라, 실제로 마주칠 상황을 기준으로 학습한다.**

## 12.3 선호 기반 보상모델: 로봇 제어에 적용하는 RLHF

시연을 구하기도 어려운 경우(예: "이 로봇 걸음걸이가 더 자연스럽다"처럼
정답 하나를 콕 집어 시연하기는 어렵지만, 두 걸음걸이를 보고 어느 쪽이
나은지 비교하기는 쉬운 경우)에는 다른 전략이 통한다 — Chapter 13에서
소개할 로봇 제어 문제에 이번 절의 아이디어를 바로 적용해볼 수 있다.
사람이 매번 "정답 행동"을 알려주는 대신, 로봇이 만든 **두 개의 시도(궤적)
중 어느 쪽이 더 나은지 고르기만** 하면 된다 — 이 아이디어는 형태만
바뀌었을 뿐, LLM을 사람이 원하는 방향으로 다듬는 RLHF(Reinforcement
Learning from Human Feedback)와 정확히 같은 구조다.

같은 상황 \\(x\\)에 대해 로봇이 만든 두 궤적 \\(y_w\\)(사람이 더
선호한 쪽)와 \\(y_l\\)(덜 선호한 쪽)이 있을 때, 보상모델
\\(r_\phi(x,y)\\)가 \\(y_w\\)에 더 높은 점수를 주도록 학습한다.
브래들리-테리 모델(Bradley-Terry model)은 이 선호 확률을 시그모이드로
모델링한다:

\\[P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\\]

이 확률을 최대화하는 손실은 다시 한번 **로지스틱회귀와 같은 교차
엔트로피 형태**다:

```python
def reward_model_loss(r_win, r_lose):
    # Bradley-Terry: P(y_w > y_l) = sigmoid(r_win - r_lose)
    return -math.log(sigmoid(r_win - r_lose))
```

이렇게 학습된 보상모델 \\(r_\phi\\)가 준비되면, 그 점수를 Chapter
9~11에서 배운 강화학습 알고리즘(특히 PPO)의 보상으로 그대로 사용해서
정책을 학습시킬 수 있다 — "사람이 직접 보상함수를 수식으로 설계하기
어려운 문제"(자연스러운 걸음걸이가 정확히 무엇인지 수식으로 정의하기는
어렵지만, 두 개를 보고 비교하기는 쉽다)에서 특히 강력한 전략이다.

## 12.4 모방학습 vs 강화학습: 언제 무엇을 쓰는가

| | 모방학습(BC/DAgger) | 강화학습(Ch.2~11) |
|---|---|---|
| 필요한 것 | 전문가 시연(또는 비교) | 보상함수, 환경과의 상호작용 |
| 탐험 필요 여부 | 기본적으로 불필요(DAgger는 약간 필요) | 필수(Chapter 2의 탐험-활용 딜레마) |
| 안전성 | 상대적으로 안전(전문가 행동만 따라함) | 학습 중 위험한 행동을 시도할 수 있음 |
| 성능 상한 | 전문가 수준이 상한(그 이상 못 넘어섬) | 이론상 전문가를 뛰어넘을 수 있음(Chapter 4의 최적 정책) |

실전에서는 둘을 섞어 쓰는 경우가 많다 — 예를 들어 행동 복제로 "그럴듯한
초기 정책"을 빠르게 만든 뒤, 그 위에서 강화학습(PPO 등)으로 미세조정해
전문가의 한계를 넘어서게 하는 식이다.

**모방학습은 "시행착오를 겪을 필요 없이, 이미 잘하는 존재를 보고 배운다"는
지름길이고, 선호 기반 보상모델은 "보상함수를 직접 설계하기 어려울 때
비교만으로 그 보상함수 자체를 학습한다"는 우회로다 — 둘 다 Chapter
2~11에서 힘들게 배운 강화학습 알고리즘 자체를 대체하는 게 아니라, 그
알고리즘에 넣을 데이터와 보상을 어떻게 더 안전하고 저렴하게 얻을
것인가에 대한 답이다.**

---

## 연습문제

**1. (코딩)** 위 `train_behavior_cloning`과 `reward_model_loss`(핵심
줄은 빈칸으로 남겨져 있다고 가정)를 완성하라:

```python
import math, random

def sigmoid(z):
    return 1 / (1 + math.exp(-max(-20, min(20, z))))

def train_behavior_cloning(demos, epochs, lr):
    # ADD ADDITIONAL CODE HERE!!
    # w, b를 0으로 초기화, epochs번 반복하며 demos의 각 (s,a)에 대해
    # 로지스틱회귀와 같은 방식으로 grad = pred - a 계산 후 w, b 갱신

    return w, b

def expert_policy(state):
    return 1 if state < 0 else 0  # 전문가: 항상 0(원점) 방향으로 이동

random.seed(0)
demos = [expert_policy(random.uniform(-5, 5)) for _ in range(200)]  # placeholder
demos = [(s, expert_policy(s)) for s in [random.uniform(-5, 5) for _ in range(200)]]
w, b = train_behavior_cloning(demos, epochs=30, lr=0.05)
print(sigmoid(w * (-2) + b) > 0.5)  # True여야 함 (state=-2에서 action=1 예측)

def reward_model_loss(r_win, r_lose):
    # ADD ADDITIONAL CODE HERE!!
    # -log(sigmoid(r_win - r_lose))

print(round(reward_model_loss(2.0, -1.0), 3))  # 0.049
```

**2. (개념 서술)** 12.2절의 복합 오차 문제가 왜 "에피소드가 길수록"
더 심각해지는지 설명하고, DAgger가 이 문제를 완화하는 메커니즘을
1~2문장으로 요약하라.

**3. (개념 서술)** 자율주행 자동차의 조향을 학습시키는 문제를 예로 들어,
(a) 순수 행동 복제만 썼을 때 발생할 수 있는 구체적인 실패 시나리오
하나, (b) 순수 강화학습(무작위 탐험 포함)만 썼을 때 발생할 수 있는
안전 문제 하나를 각각 서술하고, 왜 실전에서는 두 접근을 섞어 쓰는
경우가 많은지 논하라.
