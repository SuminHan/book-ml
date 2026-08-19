# Chapter 1. 코스 소개와 신경망 미니 리뷰 (Course Introduction & Neural Network Mini-Review)

체스나 바둑을 둘 때, "이 수가 정답이다"라고 알려주는 사람은 아무도 없다 —
게임이 끝나야 "이겼다/졌다"는 신호 하나만 돌아온다. 지금까지의 지도학습(정답
\\(y\\)가 있는 데이터)과 비지도학습(구조만 있는 데이터)과는 전혀 다른 이
문제 — **행동하고, 그 결과로 나중에 돌아오는 보상만 보고 스스로 좋은 행동을
찾아내야 하는 문제** — 가 이번 학기의 주제인 **강화학습**(Reinforcement
Learning, RL)이다. 그리고 이번 학기는 그 강화학습을, 화면 속 게임이 아니라
**물리 법칙이 지배하는 로봇 시뮬레이션**이라는 무대 위에서 다룬다.

## 1.1 강화학습이란 무엇인가: 지도학습과의 차이

| | 지도학습 | 강화학습 |
|---|---|---|
| 데이터 | 미리 주어짐, \\((x,y)\\) 쌍 | 에이전트가 행동하며 스스로 만들어냄 |
| 신호 | 정답 \\(y\\) | 보상(reward) — 얼마나 좋았는지의 점수 |
| 시간 | 각 샘플이 독립적 | 지금 행동이 미래에 보게 될 상태에 영향을 줌 |
| 새로운 딜레마 | 없음 | 탐험(exploration) vs 활용(exploitation) |

가장 중요한 차이는 **데이터가 미리 주어지지 않는다**는 것이다. 지도학습은
"이미 모아둔 사진과 라벨"로 학습하지만, 강화학습 에이전트는 지금 이 순간
어떤 행동을 하느냐에 따라 다음에 어떤 상태를 보게 될지가 달라진다 — 아직
가보지 않은 길을 시도해볼 것인가(**탐험**), 아니면 지금까지 가장 좋다고
알려진 선택을 반복할 것인가(**활용**)? 이 완전히 새로운 딜레마가 이번
학기 내내 형태를 바꿔가며 반복해서 등장한다.

## 1.2 왜 로봇 시뮬레이션인가

강화학습이 실제로 힘을 발휘하는 대표적인 무대가 로봇 제어다 — 관절에 얼마나
힘을 줄지, 어느 방향으로 움직일지를 사람이 규칙으로 일일이 정해주는 대신,
시행착오를 통해 "넘어지지 않고 걷는 법"을 스스로 찾아내게 하는 것이다. 이번
학기는 게임 화면 같은 추상적인 예제뿐 아니라, 실제로 물리 엔진 위에서 로봇을
움직여보는 시뮬레이션 환경(Chapter 13~14에서 본격적으로 다룬다)을 실습에
사용한다.

## 1.3 이번 학기 로드맵

- **Chapter 2~7 (Block A)**: 강화학습의 이론적 뼈대를 표(table) 기반으로
  차근차근 쌓는다 — 멀티암 밴딧에서 시작해, MDP, 동적계획법, 몬테카를로,
  시간차 학습, n-step/적격흔적까지. 이 순서는 강화학습의 표준 교과서인
  Sutton과 Barto의 *Reinforcement Learning: An Introduction*의 목차를
  그대로 따른다.
- **Chapter 8**: Block A 팀 프로젝트와 중간고사 준비.
- **Chapter 9~11 (Block B 전반)**: 표 대신 신경망으로 확장하는 심층강화학습
  (DQN)과, Q값을 거치지 않고 정책을 직접 학습하는 정책기반 방법(REINFORCE,
  PPO)까지.
- **Chapter 12**: 모방학습과 인간 피드백 — 시행착오 대신 시연이나 선호로부터
  배우는 법.
- **Chapter 13~14**: 로봇 시뮬레이션과 제어의 기초, 그리고 더 정교한
  물리엔진(MuJoCo)과 GPU 가속 시뮬레이션(NVIDIA Isaac Sim)까지.
- **Chapter 15**: 환경의 모델을 직접 활용하는 모델기반 RL과, 바둑을 정복한
  AlphaGo의 핵심 아이디어인 몬테카를로 트리 탐색(MCTS).
- **Chapter 16**: Block B 팀 프로젝트와 학기 총정리.

## 1.4 이 과목은 ML1과 완전히 독립적이다

ML1을 듣지 않았어도 이 과목을 바로 시작할 수 있다 — 이번 학기가 필요로 하는
신경망 지식은 순전파, 역전파, 그리고 경사하강법으로 손실을 줄인다는 감각
정도이며, 아래 1.5절에서 그 핵심만 압축해서 다시 짚는다. (ML1을 먼저 들었다면
익숙한 내용이니 빠르게 훑고 넘어가도 좋다.)

## 1.5 신경망 미니 리뷰

신경망은 입력 \\(x\\)를 층(layer)을 거치며 변환해 예측값을 만드는 함수다.
2층 신경망의 순전파:

\\[z_1 = W_1 x + b_1, \quad a_1 = \sigma(z_1), \quad z_2 = W_2^T a_1 + b_2,
\quad a_2 = \sigma(z_2)\\]

**학습**이란 예측이 얼마나 틀렸는지를 손실함수 \\(J\\)로 수치화하고,
\\(J\\)를 줄이는 방향(그래디언트의 반대 방향)으로 파라미터
\\(W_1, W_2, b_1, b_2\\)를 조금씩 옮기는 과정이다. 그 그래디언트를 구하는
방법이 **역전파**(backpropagation) — 연쇄법칙을 출력에서 입력 방향으로
거꾸로 적용해, 각 파라미터가 최종 손실에 얼마나 책임이 있는지 계산하는
것이다. 이번 학기에서는 PyTorch의 `nn.Module`과 `.backward()`로 이 계산을
자동으로 처리한다 — 손으로 유도하는 연습은 이미 (ML1을 들었다면) 끝냈다고
가정한다.

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

# 간단한 회귀 실습: y = 2x + 1을 배우게 하기
torch.manual_seed(0)
model = MLP(1, 16, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

X = torch.rand(64, 1) * 4 - 2
y = 2 * X + 1

for epoch in range(200):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("최종 손실:", round(loss.item(), 4))
print("x=1일 때 예측:", round(model(torch.tensor([[1.0]])).item(), 3), "(정답: 3.0)")
```

## 1.6 실습 환경 세팅: Gymnasium

이번 학기 실습은 **Gymnasium**(옛 OpenAI Gym의 후신)이라는 표준 라이브러리로
만든 환경 위에서 진행한다 — 어떤 환경이든 "현재 상태를 관측하고(`reset`),
행동을 하나 취하면(`step`) 다음 상태·보상·종료 여부가 돌아온다"는 똑같은
인터페이스를 따른다. 이 인터페이스 통일 덕분에, Chapter 2~11에서 배우는
어떤 알고리즘이든 환경만 바꿔 끼우면 그대로 재사용할 수 있다.

```python
import gymnasium as gym

env = gym.make("CartPole-v1")  # 막대를 쓰러뜨리지 않고 균형 잡기
obs, info = env.reset(seed=0)
print("초기 상태(카트 위치, 속도, 막대 각도, 각속도):", obs)

for _ in range(3):
    action = env.action_space.sample()  # 지금은 무작위 정책
    obs, reward, terminated, truncated, info = env.step(action)
    print("행동:", action, "-> 다음 상태:", obs, "보상:", reward)

env.close()
```

이 CartPole 환경은 이번 학기 초반의 표 기반 알고리즘을 시각적으로 이해하는
데, 그리고 Chapter 9의 DQN에서 다시 등장한다.

**강화학습은 지도학습·비지도학습과 완전히 다른 게임이다 — 정답을 맞히는
것도, 구조를 찾는 것도 아니라, 스스로 행동해서 얻은 경험만으로 무엇이 좋은
행동인지 알아내야 한다. 이 게임의 규칙을 정확히 정의하는 것부터가 다음
장의 시작이다.**

---

## 연습문제

**1. (코딩)** 위 `MLP` 클래스와 학습 루프를 참고해서, `y = x^2` 함수를
근사하는 신경망을 학습시켜라(`hidden_dim=32`, 200 에포크 이상 학습한 뒤
`x=1.5`에서의 예측값이 실제 값 `2.25`에 충분히 가까운지 확인하라).

**2. (개념 서술)** 지도학습에서는 "탐험과 활용의 딜레마"가 왜 존재하지
않는지, 그리고 강화학습에서는 왜 이 딜레마가 근본적으로 피할 수 없는지
1.1절의 표를 참고해서 두세 문장으로 설명하라.

**3. (실습)** 위 Gymnasium 코드를 그대로 실행한 뒤, `env.action_space`와
`env.observation_space`를 각각 출력해서 CartPole의 행동이 몇 가지이고
(이산/연속 중 무엇인지), 상태가 몇 차원 벡터인지 확인하라. `Pendulum-v1`
환경으로 바꿔서 같은 것을 확인하고, CartPole과 행동 공간의 형태가 어떻게
다른지(이산 vs 연속) 한 문장으로 비교하라.
