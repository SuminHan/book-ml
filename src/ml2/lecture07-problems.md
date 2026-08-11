# Problem Set

난이도 등급: **Tier C (폴백 준비 대상)** — 아래 두 버전을 모두 준비해두고, 학생 반응을
본 뒤 선택한다.

**1.** (코딩) 다음과 같은 `ReplayBuffer` 클래스를 완성하라:

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
print(buf.buffer[0])    # ("s2","a2",0,"s3") -- ("s1",...)는 이미 제거됨
```

**2.** (코딩) 타겟 네트워크 갱신 주기를 시뮬레이션하는 `should_update_target`을
작성하라: 현재 학습 스텝 `step`이 `update_freq`의 배수일 때만 `True`를 반환한다.

```python
def should_update_target(step, update_freq):
    # ADD ADDITIONAL CODE HERE!!

print([should_update_target(s, 1000) for s in [999, 1000, 1500, 2000]])
# [False, True, False, True]
```

---

## 손유도 과제 — 두 가지 버전 중 택1 (교원 판단)

### [버전 A] 자유 논증 (Math for ML이 최적화의 수렴 개념을 다뤘을 경우)

타겟 네트워크 없이(즉 목표값 계산에도 \\(\theta\\)를 그대로 쓴다고 하자) DQN을
학습시킨다고 하자. 손실 \\(J(\theta) = (r + \gamma \max_{a'} Q(s',a';\theta) -
Q(s,a;\theta))^2\\)를 \\(\theta\\)로 미분할 때, \\(\theta\\)가 목표값 항
\\(Q(s',a';\theta)\\)와 예측값 항 \\(Q(s,a;\theta)\\) **둘 다**에 나타난다는 것을
지적하고, 이 때문에 그래디언트 업데이트 한 번이 "예측값을 목표값에 가깝게" 옮길
뿐 아니라 "목표값 자체도 함께" 움직이게 만든다는 것을 논증하라. 그런 다음, 타겟
네트워크(\\(\theta^-\\)를 목표값 계산에만 쓰고, 일정 주기로만 갱신)가 이 문제를 어떻게
피하는지 설명하라.

### [버전 B] 구조화된 논증 워크시트 (폴백)

```
타겟 네트워크가 없을 때:
  J(theta) = (r + gamma * max_a' Q(s',a';theta) - Q(s,a;theta))^2
  이 식에서 theta는 몇 군데에 등장하는가? ______________
  (힌트: max_a' Q(s',a';theta) 안에도, Q(s,a;theta) 안에도 등장한다)

  theta를 한 번 업데이트하면:
  - 예측값 Q(s,a;theta)는 목표값에 ______________ (가까워진다/멀어진다)
  - 그런데 목표값 자체(max_a' Q(s',a';theta))도 ______________ (같이 바뀐다/그대로다)
  - 결과: "쫓아가려는 대상이 ______________" 상황이 반복된다

타겟 네트워크(theta^-)가 있을 때:
  J(theta) = (r + gamma * max_a' Q(s',a';theta^-) - Q(s,a;theta))^2
  theta^-는 매 스텝 ______________ (바뀐다/고정돼 있다)
  따라서 theta를 업데이트하는 동안, 목표값은 ______________ (움직인다/고정돼 있다)
```

**정확성 확인**: 타겟 네트워크의 갱신 주기(`update_freq`)를 극단적으로 크게(예:
100만 스텝) 잡으면 어떤 문제가 생길지, 반대로 극단적으로 작게(예: 1 스텝, 사실상
타겟 네트워크가 없는 것과 같음) 잡으면 어떤 문제가 생길지 각각 한 문장으로
설명하라.

---

*교원 노트: 버전 A/B 중 선택은 Math for ML의 최적화 개념 커버리지 확인 후 결정.
확인 전까지는 버전 B를 기본값으로 준비.*
