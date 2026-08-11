# Topics Covered

## Q-함수를 신경망으로 근사

\\(Q(s,a)\\)를 테이블 대신 파라미터 \\(\theta\\)를 가진 신경망 \\(Q(s,a;\theta)\\)로
표현한다. 손실함수는 W06의 TD 오차를 제곱한 것이다:

\\[J(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s',a';\theta) -
Q(s,a;\theta)\right)^2\right]\\]

이 손실을 최소화하도록 경사하강법(정확히는 W07의 역전파)으로 \\(\theta\\)를
갱신한다 — 지도학습과 형태는 같지만, "정답"에 해당하는 목표값
\\(r + \gamma \max_{a'} Q(s',a';\theta)\\) 자체가 지금 학습 중인 \\(\theta\\)로
계산된다는 점이 다르다.

## 문제 1: 목표가 계속 움직인다

\\(\theta\\)가 매 스텝 갱신되면, 손실함수의 목표값도 매 스텝 바뀐다 — 지도학습이라면
정답 \\(y\\)는 절대 안 바뀌는데, 여기선 "정답"이 스스로를 쫓아다니는 셈이다. 이는
학습을 불안정하게 만든다(진동, 발산).

**해결책: 타겟 네트워크(Target Network)**. \\(Q\\)와 똑같은 구조를 가진 두 번째
신경망 \\(Q(s,a;\theta^-)\\)를 두고, 목표값 계산에는 **이 타겟 네트워크만** 쓴다:

\\[J(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s',a';\theta^-) -
Q(s,a;\theta)\right)^2\right]\\]

\\(\theta^-\\)는 매 스텝 갱신되지 않고, 일정 주기마다(예: 1000 스텝마다)
\\(\theta\\)의 현재 값으로 통째로 복사된다. 그 사이 구간 동안 목표값은 고정돼
있으므로, 손실함수는 (짧은 구간 동안은) 지도학습과 똑같이 "정답이 고정된" 최적화
문제가 된다 — 움직이는 과녁을 고정시킨 셈이다.

```python
def dqn_loss(Q_net, target_net, s, a, r, s_next, gamma, actions):
    target = r + gamma * max(target_net(s_next, a2) for a2 in actions)
    prediction = Q_net(s, a)
    return (target - prediction) ** 2
```

## 문제 2: 데이터가 서로 상관돼 있다

연속된 프레임(상태)은 서로 거의 같아서, 순서대로 학습하면 최근 몇 개의 비슷한
경험에 과도하게 치우친(overfitting) 학습이 된다 — 게다가 신경망 학습은 미니배치가
서로 독립적이고 다양할수록 안정적이라는 가정에 기대고 있다.

**해결책: 경험 재현(Experience Replay)**. 에이전트가 경험한 \\((s, a, r, s')\\)를
바로 학습에 쓰지 않고, 일단 **재현 버퍼(replay buffer)**라는 큰 저장소에 쌓아둔다.
학습할 때는 이 버퍼에서 **무작위로** 미니배치를 샘플링해서 쓴다 — 이러면 학습 배치가
시간적으로 인접한 경험들로만 구성되는 것을 피하고, 오래된 경험도 다시 재사용할 수
있어 데이터 효율도 좋아진다.

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

## 두 장치의 공통점

경험 재현과 타겟 네트워크는 서로 다른 문제(데이터 상관성 vs 움직이는 목표)를
풀지만, 근본적으로 같은 철학을 공유한다: **"지도학습이 잘 작동하는 조건(독립적인
데이터, 고정된 정답)을, 강화학습이라는 원래 그 조건이 깨진 환경에서 인위적으로
다시 만들어준다."**
