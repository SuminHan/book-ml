# Problem Set

난이도 등급: **Tier C — 최우선 폴백 대상** (Fable 검토: "대학원생도 버거운 난이도")

**1.** (코딩) 간단한 이산 행동공간(2개 행동) softmax 정책에 대해, 한 에피소드의
REINFORCE 업데이트를 구현하라.

```python
import math

def softmax_policy(theta, state_feature):
    # theta: 파라미터(길이2), state_feature: 스칼라(단순화를 위해 상태를 스칼라 하나로 표현)
    # ADD ADDITIONAL CODE HERE!!
    # logits = [theta[0]*state_feature, theta[1]*state_feature]
    # softmax 확률 계산 (수치안정성을 위해 max 빼고 exp)
    return probs

def reinforce_update(theta, episode, alpha, gamma):
    # episode: [(state_feature, action, reward), ...]
    # ADD ADDITIONAL CODE HERE!!
    # 리턴(return) G_t를 뒤에서부터 누적 계산 (할인 적용)
    for t, (s, a, r) in enumerate(episode):
        probs = softmax_policy(theta, s)
        # ADD ADDITIONAL CODE HERE!!
        # grad_log_pi: action=0이면 [1-probs[0], -probs[1]]*s, action=1이면 [-probs[0], 1-probs[1]]*s
        # theta 갱신: theta += alpha * G[t] * grad_log_pi
    return theta
```

---

## 손유도 과제 — 두 가지 버전 중 택1 (교원 판단, 기본값 = 버전 B)

### [버전 A] Policy Gradient Theorem 완전 유도 (심화반/수학 상위권 전용)

정책 \\(\pi_\theta\\) 하에서 기대 리턴 \\(J(\theta) = \mathbb{E}_{\tau \sim
\pi_\theta}[R(\tau)]\\)의 그래디언트가

\\[\nabla_\theta J(\theta) = \mathbb{E}_{\tau}\left[\sum_t \nabla_\theta \log
\pi_\theta(a_t|s_t) \, G_t\right]\\]

가 됨을, **로그미분 트릭**(log-derivative trick) \\(\nabla_\theta \pi =
\pi \nabla_\theta \log \pi\\)을 사용하여 처음부터 끝까지 유도하라.

### [버전 B] 빈칸채움형 유도 워크시트 (기본값 — 대부분 학생 대상)

핵심 트릭 하나만 이해하면 되도록 구조를 다 주고, 빈칸만 채우게 한다:

```
목표: J(theta) = sum_tau P(tau; theta) * R(tau) 를 theta로 미분하고 싶다.
문제: P(tau; theta)를 직접 미분하면 기댓값(적분) 형태가 깨져서 샘플로 추정할 수 없다.

로그미분 트릭: grad(f) = f * grad(log f)  [분수미분 공식에서 유도됨: grad(log f) = grad(f)/f]

Step 1: grad_theta P(tau;theta) = P(tau;theta) * ______________  [로그미분 트릭 적용]

Step 2: grad_theta J(theta) = sum_tau ______________ * R(tau)
                             = sum_tau P(tau;theta) * grad_theta log P(tau;theta) * R(tau)
                             = E_tau[ grad_theta log P(tau;theta) * R(tau) ]   <- 다시 기댓값 형태!

Step 3: 궤적 확률 P(tau;theta) = prod_t pi_theta(a_t|s_t) * (환경전이확률, theta와 무관)
        따라서 log P(tau;theta) = sum_t log pi_theta(a_t|s_t) + (theta 무관 상수항)
        => grad_theta log P(tau;theta) = ______________

결론: grad_theta J(theta) = E_tau[ (sum_t grad_theta log pi_theta(a_t|s_t)) * R(tau) ]
```

**정확성 확인**: Step 3의 "환경전이확률은 theta와 무관하다"는 사실이 왜 중요한지
한 문장으로 설명하라 (힌트: 이게 없으면 환경 모델을 몰라도 정책만으로 학습
가능하다는 model-free RL의 핵심 성질이 깨진다).

---

*교원 노트: 버전 A는 수학 상위권/희망자 대상 선택 심화로만 운영 권장. 전체 학생
기본값은 버전 B(워크시트) — Fable 검토에서 "PG Theorem 완전유도는 최우선 폴백
대상"으로 지적됨.*
