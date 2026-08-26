# Chapter 15. 잠재변수 생성모델: VAE에서 GAN, Diffusion까지 (Latent-Variable Generative Models)

[![Open In Colab: VAE](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml1/chapter15_vae_elbo.ipynb)
[![Open In Colab: GAN](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml1/chapter15_gan.ipynb)

[STUB: cq가 확장할 자리 — 아래는 핵심 뼈대(15.1 EM/GMM 문단을 4.4로
옮긴 자리에 짧은 다리 문단으로 교체, 15.3→15.2 리넘버링 반영, GAN은
그대로 유지, Diffusion에 Waymo SceneDiffuser 실전 사례 추가).]

4.4절에서 본 **EM**(Expectation-Maximization) 알고리즘은 "관측 안 된
잠재변수가 데이터를 어떻게 만들어내는가"라는 질문에 대한 가장 고전적인
답이었다 — 이 장은 그 질문을 신경망으로 확장한다. VAE[^vae]는 EM과
정확히 같은 질문("잠재변수 \\(z\\)가 데이터 \\(x\\)를 어떻게
만들어내는가")을 던지지만, 계산 불가능한 E-step을 신경망 인코더로
대체한다. GAN·Diffusion은 완전히 다른 두 원리(적대적 학습, 점진적
노이즈 제거)로 같은 목표에 도달한다 — 생성형 모델의 세 가지 얼굴을
이 장에서 훑는다.

이 질문은 Chapter 14 "표현학습"의 연장선이다. 거기서 우리는 "정보를 담는
벡터로 데이터를 압축한다"는 직관을 배웠다 — PCA의 주성분, word2vec[^word2vec]의 단어
벡터. 이번 장은 그 압축된 표현에 **확률**을 입힌다. 잠재변수 \\(z\\)는
단 하나의 점사상이 아니라 데이터의 "설명"이 되고, 질문은 둘로 갈라진다 —
"관측된 \\(x\\)로부터 보이지 않는 \\(z\\)를 어떻게 추론할 것인가", 그리고
"알려진 분포에서 \\(z\\)를 뽑아 어떻게 새로운 \\(x\\)를 만들어낼
것인가". VAE의 인코더가 \\(z\\) 위의 한 점이 아니라 **분포**를 내놓는
것은, Chapter 14의 압축 직관을 딥러닝 버전으로 확률화한 모습이다. 반대로
이 장에서 배울 세 원리(VAE·GAN·Diffusion)는 다음 장인 "Block B
캡스톤: 팀 프로젝트와 ML1 총정리"(Chapter 16)에서 팀 프로젝트의 모델을
고를 때 안정성·생성 속도·품질의 기준으로 선택하고 정당화할 수 있는
도구 상자가 된다.[^cs230]

이 챕터를 마치면 다음을 할 수 있다:

- 계산 불가능한 우도를 대체하는 계산 가능한 하한 ELBO를 복원 항과 KL
  정규화 항 두 부분으로 분해하고, 직접 학습한 VAE의 손실 곡선에서 두 항의
  줄다리기가 균형점에 이르는 과정을 읽을 수 있다.
- GAN의 min-max 게임을 내쉬 균형으로 설명하고, "판별자 정확도 50%가 목표
  운영점"이라는 실전 판독과, 목적함수에 다양성 항이 없어 생기는 모드
  붕괴의 원인을 1차원 장난감 예제로 진단할 수 있다.
- VAE·GAN·Diffusion 세 원리를 잠재변수의 종류, 학습 방법, 안정성,
  생성 속도의 세 축으로 비교하고, Stable Diffusion이 VAE의 압축된 잠재
  공간에서 Diffusion을 돌리는 구조임을, 그리고 이미지 생성 밖의 응용
  (자율주행 시뮬레이션 등)까지 설명할 수 있다.

이번 주는 두 개의 수업 블록으로 진행된다:

- [15.1 VAE: ELBO와 실습](chapter15/1.md) — 잠재변수를 연속 벡터로,
  계산 불가능한 E-step을 신경망으로 대체한 VAE[^vae]를 배운다. 옌센 부등식 한
  번으로 ELBO를 유도하고 복원 항·KL 정규화 항의 의미를 확인한 뒤, 두
  봉우리 데이터로 작은 VAE를 실제로 학습해 표준정규분포에서 \\(z\\)를 뽑아
  디코더만 통과시키는 생성 모드까지 확인한다(리파라미터화 트릭, β-VAE[^beta_vae],
  "흐릿한" 생성 이미지의 원인 포함).
- [15.2 GAN, Diffusion, 그리고 세 원리 비교](chapter15/2.md) — 우도
  기반과 다른 두 패러다임을 만난다. GAN[^gan]은 생성자·판별자의 min-max 게임
  (내쉬 균형, 1차원 장난감의 균형점 계산, 모드 붕괴, WGAN[^wgan]·CycleGAN[^cyclegan]·
  StyleGAN[^stylegan]까지)이고, Diffusion은 노이즈를 한 단계씩 제거하는 MSE 회귀
  문제(DDPM[^ddpm]/DDIM[^ddim], VAE 잠재 공간을 쓰는 Stable Diffusion[^stablediffusion],
  그리고 이미지 밖의 응용 사례로 자율주행 시뮬레이션 시나리오를 생성하는
  Waymo SceneDiffuser[^scenediffuser])다. 마지막으로 세 원리를
  잠재변수/학습 방법/안정성/생성 속도로 나란히 비교해 장을 마무리한다.

[^word2vec]: Mikolov, T., Chen, K., Corrado, G., Dean, J. (2013). "Efficient Estimation of Word Representations in Vector Space." arXiv:1301.3781.
[^vae]: Kingma, D. P., Welling, M. (2013). "Auto-Encoding Variational Bayes." arXiv:1312.6114.
[^gan]: Goodfellow, I. J. et al. (2014). "Generative Adversarial Networks." arXiv:1406.2661.
[^ddpm]: Ho, J., Jain, A., Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS 2020. arXiv:2006.11239.
[^stablediffusion]: Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022. arXiv:2112.10752.
[^beta_vae]: Higgins, I., Matthey, L., Pal, A., et al. (2017). "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework." ICLR 2017. arXiv:1806.03916.
[^ddim]: Song, J., Meng, C., Ermon, S. (2021). "Denoising Diffusion Implicit Models." ICLR 2021. arXiv:2010.02502.
[^wgan]: Arjovsky, M., Chintala, S., Bottou, L. (2017). "Wasserstein GAN." arXiv:1701.07875.
[^stylegan]: Karras, T., Laine, S., Aila, T. (2018). "A Style-Based Generator Architecture for Generative Adversarial Networks." arXiv:1812.04948.
[^cs230]: Stanford CS230: Deep Learning. https://cs230.stanford.edu/
[^cyclegan]: Zhu, J.-Y., Park, T., Isola, P., Efros, A. A. (2017). "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." CVPR 2017. arXiv:1703.10593.
[^scenediffuser]: Jiang, C. M. et al. (2024). "SceneDiffuser: Efficient and Controllable Driving Simulation Initialization and Rollout." NeurIPS 2024.
