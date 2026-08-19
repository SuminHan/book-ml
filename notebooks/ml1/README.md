# ML1 실습 노트북

책의 각 챕터에 나오는 코드를 실제로 실행해볼 수 있는 Google Colab
노트북입니다. 각 책 챕터 페이지 상단의 "Open in Colab" 배지를 눌러도
바로 열립니다.

| 챕터 | 노트북 | 내용 |
|---|---|---|
| [Ch15](https://smhanlab.com/book-ml/kor/ml1/chapter15.html) | [chapter15_vae_elbo.ipynb](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml1/chapter15_vae_elbo.ipynb) | KL divergence, VAE 손실, 정규화 항 시각화 |
| [Ch15](https://smhanlab.com/book-ml/kor/ml1/chapter15.html) | [chapter15_gan.ipynb](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml1/chapter15_gan.ipynb) | 판별자/생성자 손실, 내쉬 균형(D=0.5) 확인 |

2026-08 개편으로 VAE·GAN이 ML2에서 ML1 Chapter 15(잠재변수
생성모델: EM/GMM에서 VAE, GAN, Diffusion까지)로 옮겨오면서, 이
두 노트북도 `notebooks/ml2/`에서 여기로 이동했습니다.

## 만든 방식

각 노트북은 순수 Python(numpy/matplotlib만 사용, 딥러닝 프레임워크
없음)으로, 책 본문에 있는 함수를 그대로 가져와 실제로 실행하고, 책의
연습문제에 있는 기댓값과 대조하는 assert를 포함합니다 — 전부 로컬에서
`python`으로 먼저 실행 검증한 뒤 커밋했습니다.

주석/설명은 한국어입니다. 코드 자체는 언어 무관이라 영어판 책
챕터에서도 같은 노트북으로 링크합니다.
