# ML2 실습 노트북

책의 각 챕터에 나오는 코드를 실제로 실행해볼 수 있는 Google Colab
노트북입니다. 각 책 챕터 페이지 상단의 "Open in Colab" 배지를 눌러도
바로 열립니다.

| 챕터 | 노트북 | 내용 |
|---|---|---|
| [Ch5](https://smhanlab.com/book-ml/kor/ml2/chapter05.html) | [chapter05_policy_evaluation.ipynb](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter05_policy_evaluation.ipynb) | 반복적 정책평가, 벨만방정식 직접 검증 |
| [Ch6](https://smhanlab.com/book-ml/kor/ml2/chapter06.html) | [chapter06_q_learning.ipynb](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter06_q_learning.ipynb) | 장난감 환경에서 Q-learning이 최적 정책을 찾는 과정 |
| [Ch7](https://smhanlab.com/book-ml/kor/ml2/chapter07.html) | [chapter07_dqn_tricks.ipynb](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter07_dqn_tricks.ipynb) | 경험 재현 버퍼, 타겟 네트워크 동기화 |
| [Ch8](https://smhanlab.com/book-ml/kor/ml2/chapter08.html) | [chapter08_reinforce_ppo.ipynb](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter08_reinforce_ppo.ipynb) | REINFORCE 학습 곡선, PPO 클리핑 목적함수 시각화 |
| [Ch10](https://smhanlab.com/book-ml/kor/ml2/chapter10.html) | [chapter10_vae_elbo.ipynb](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter10_vae_elbo.ipynb) | KL divergence, VAE 손실, 정규화 항 시각화 |
| [Ch11](https://smhanlab.com/book-ml/kor/ml2/chapter11.html) | [chapter11_gan.ipynb](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter11_gan.ipynb) | 판별자/생성자 손실, 내쉬 균형(D=0.5) 확인 |

## 만든 방식

각 노트북은 순수 Python(numpy/matplotlib만 사용, 딥러닝 프레임워크
없음)으로, 책 본문에 있는 함수를 그대로 가져와 실제로 실행하고, 책의
연습문제에 있는 기댓값과 대조하는 assert를 포함합니다 — 전부 로컬에서
`python`으로 먼저 실행 검증한 뒤 커밋했습니다.

주석/설명은 한국어입니다. 코드 자체는 언어 무관이라 영어판 책
챕터에서도 같은 노트북으로 링크합니다.
