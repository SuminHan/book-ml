# Chapter 7. 심층강화학습 (Deep Reinforcement Learning)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter07_dqn_tricks.ipynb)

2013년, 딥마인드는 Q-learning의 Q-테이블을 신경망으로 통째로 바꾼
알고리즘으로 Atari 2600 게임 여러 개를 학습시켰다 — 게임의 규칙을 전혀
알려주지 않고, 오직 화면 픽셀과 점수만 보고서. **DQN**(Deep
Q-Network)이라 불린 이 결과는 2015년 Nature에 게재되며, 여러 게임에서
사람 수준 또는 그 이상의 성능을 보였다.

## 7.1 Q-테이블이 감당 못 하는 규모

Chapter 6의 Q-learning은 `Q[s][a]`라는 **표**(테이블)에 모든 상태-행동
쌍의 값을 저장했다. 이건 상태가 몇백 개 수준일 때는 잘 작동하지만,
Atari 게임의 화면(가로세로 수백 픽셀, 색상 조합)이 만들어낼 수 있는
상태의 가짓수는 사실상 무한에 가깝다 — 테이블에 다 담을 수도 없고,
대부분의 상태는 학습 중에 단 한 번도 안 나타난다.

## 7.2 Q-함수를 신경망으로 근사

DQN의 아이디어는 단순하다: 테이블 대신 **신경망**으로 \\(Q(s,a)\\)를
근사하자(\\(Q(s,a;\theta)\\), \\(\theta\\)는 신경망 가중치). 비슷한
화면(상태)은 신경망을 통해 비슷한 Q값을 내게 되므로, 한 번도 정확히 본
적 없는 상태에도 일반화(generalize)할 수 있다. 손실함수는 Chapter 6의
TD 오차를 제곱한 것이다:

\\[J(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s',a';\theta) -
Q(s,a;\theta)\right)^2\right]\\]

이 손실을 최소화하도록 경사하강법(정확히는 역전파)으로 \\(\theta\\)를
갱신한다 — 지도학습과 형태는 같지만, "정답"에 해당하는 목표값
\\(r + \gamma \max_{a'} Q(s',a';\theta)\\) 자체가 지금 학습 중인
\\(\theta\\)로 계산된다는 점이 다르다.

## 7.3 문제 1: 목표가 계속 움직인다

\\(\theta\\)가 매 스텝 갱신되면, 손실함수의 목표값도 매 스텝 바뀐다 —
지도학습이라면 정답 \\(y\\)는 절대 안 바뀌는데, 여기선 "정답"이
스스로를 쫓아다니는 셈이다. 이는 학습을 불안정하게 만든다(진동, 발산).

**해결책: 타겟 네트워크**(Target Network). \\(Q\\)와 똑같은 구조를
가진 두 번째 신경망 \\(Q(s,a;\theta^-)\\)를 두고, 목표값 계산에는
**이 타겟 네트워크만** 쓴다:

\\[J(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s',a';\theta^-) -
Q(s,a;\theta)\right)^2\right]\\]

\\(\theta^-\\)는 매 스텝 갱신되지 않고, 일정 주기마다(예: 1000
스텝마다) \\(\theta\\)의 현재 값으로 통째로 복사된다. 그 사이 구간
동안 목표값은 고정돼 있으므로, 손실함수는(짧은 구간 동안은) 지도학습과
똑같이 "정답이 고정된" 최적화 문제가 된다 — 움직이는 과녁을 고정시킨
셈이다.

```python
def dqn_loss(Q_net, target_net, s, a, r, s_next, gamma, actions):
    target = r + gamma * max(target_net(s_next, a2) for a2 in actions)
    prediction = Q_net(s, a)
    return (target - prediction) ** 2
```

## 7.4 문제 2: 데이터가 서로 상관돼 있다

연속된 프레임(상태)은 서로 거의 같아서, 순서대로 학습하면 최근 몇 개의
비슷한 경험에 과도하게 치우친(overfitting) 학습이 된다 — 게다가 신경망
학습은 미니배치가 서로 독립적이고 다양할수록 안정적이라는 가정에 기대고
있다.

**해결책: 경험 재현**(Experience Replay). 에이전트가 경험한
\\((s, a, r, s')\\)를 바로 학습에 쓰지 않고, 일단 **재현 버퍼**(replay
buffer)라는 큰 저장소에 쌓아둔다. 학습할 때는 이 버퍼에서 **무작위로**
미니배치를 샘플링해서 쓴다 — 이러면 학습 배치가 시간적으로 인접한
경험들로만 구성되는 것을 피하고, 오래된 경험도 다시 재사용할 수 있어
데이터 효율도 좋아진다.

```python
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

## 7.5 두 장치의 공통점

경험 재현과 타겟 네트워크는 서로 다른 문제(데이터 상관성 vs 움직이는
목표)를 풀지만, 근본적으로 같은 철학을 공유한다: **"지도학습이 잘
작동하는 조건(독립적인 데이터, 고정된 정답)을, 강화학습이라는 원래 그
조건이 깨진 환경에서 인위적으로 다시 만들어준다."**

**"강화학습에 신경망을 붙이면 저절로 잘 될 것"이라는 순진한 기대는
틀렸다 — DQN의 진짜 기여는 신경망 자체가 아니라, 그 조합이 안정적으로
학습되도록 만든 몇 가지 공학적 장치에 있다.**

---

## 연습문제

**1. (코딩)** 다음과 같은 `ReplayBuffer` 클래스와
`should_update_target`을 완성하라(핵심 줄은 빈칸으로 남겨져 있다고
가정):

```python
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition):
        # ADD ADDITIONAL CODE HERE!!
        # 버퍼가 capacity를 넘으면 가장 오래된 항목을 제거(FIFO)한 뒤 추가

    def sample(self, batch_size):
        # ADD ADDITIONAL CODE HERE!!
        # 버퍼에서 batch_size개를 무작위 비복원추출로 반환

buf = ReplayBuffer(capacity=3)
buf.push(("s1","a1",1,"s2"))
buf.push(("s2","a2",0,"s3"))
buf.push(("s3","a3",1,"s4"))
buf.push(("s4","a4",0,"s5"))  # 버퍼가 가득 차서 첫 번째 항목이 밀려남
print(len(buf.buffer))  # 3
print(buf.buffer[0])    # ("s2","a2",0,"s3")

def should_update_target(step, update_freq):
    # ADD ADDITIONAL CODE HERE!!

print([should_update_target(s, 1000) for s in [999, 1000, 1500, 2000]])
# [False, True, False, True]
```

**2. (손유도, Tier C — 폴백 준비 대상)** 타겟 네트워크 없이(즉 목표값
계산에도 \\(\theta\\)를 그대로 쓴다고 하자) DQN을 학습시킨다고 하자.
손실 \\(J(\theta) = (r + \gamma \max_{a'} Q(s',a';\theta) -
Q(s,a;\theta))^2\\)에서 \\(\theta\\)가 목표값 항 \\(Q(s',a';\theta)\\)와
예측값 항 \\(Q(s,a;\theta)\\) **둘 다**에 나타난다는 것을 지적하고,
이 때문에 그래디언트 업데이트 한 번이 "예측값을 목표값에 가깝게" 옮길
뿐 아니라 "목표값 자체도 함께" 움직이게 만든다는 것을 논증하라. 그런
다음, 타겟 네트워크(\\(\theta^-\\)를 목표값 계산에만 쓰고, 일정
주기로만 갱신)가 이 문제를 어떻게 피하는지 설명하라.

**빈칸채움형 폴백 버전** (자유 논증이 어려운 경우):

```
타겟 네트워크가 없을 때:
  theta는 max_a' Q(s',a';theta) 안에도, Q(s,a;theta) 안에도 등장한다.
  theta를 한 번 업데이트하면 예측값은 목표값에 ______________ (가까워진다/멀어진다)
  그런데 목표값 자체도 ______________ (같이 바뀐다/그대로다)

타겟 네트워크(theta^-)가 있을 때:
  theta^-는 매 스텝 ______________ (바뀐다/고정돼 있다)
  따라서 theta를 업데이트하는 동안, 목표값은 ______________ (움직인다/고정돼 있다)
```

**정확성 확인**: 타겟 네트워크의 갱신 주기(`update_freq`)를 극단적으로
크게(예: 100만 스텝) 잡으면 어떤 문제가 생길지, 반대로 극단적으로
작게(예: 1 스텝) 잡으면 어떤 문제가 생길지 각각 한 문장으로 설명하라.
