# Chapter 16. Block B 캡스톤: 팀 프로젝트와 ML1 총정리 (Block B Capstone & ML1 Review)

Chapter 9~15는 신경망 기초에서 시작해 CNN[^alexnet], 시퀀스 모델, Transformer[^transformer], LLM[^gpt3],
그리고 잠재변수 생성모델까지 현대 딥러닝의 핵심 계보를 훑었다. 이번 장은
Chapter 8의 짝이다 — 새 개념 대신, 이 두 번째 절반의 도구들을 실제 데이터에
적용해보고 학기를 총정리하는 자리다.

**이 순서가 왜 여기인가.** Chapter 15에서 EM/GMM[^gmm]에서 출발해 VAE[^vae], GAN[^gan],
Diffusion[^ddpm]까지 — 관측되지 않은 잠재변수로 데이터를 만들어내는 생성모델의
네 가지 얼굴까지 손을 잡았다. Block B의 도구상자는 이제 채워진 셈이다.
Block A(Chapter 2~7)가 Chapter 8에서 "정형 데이터를 다루는 고전 ML의
완결"을 검증했다면, 이번 캡스톤은 그와 같은 구조로 Block B의 도구를 실제
데이터 앞에 세운다. 다만 검증의 무게는 다르다. Chapter 8이 "배운 도구가
이 데이터에 **맞는가**"를 물었다면, 이번 프로젝트는 "이 데이터에
**딥러닝이 필요한가**"를 — 정형 데이터의 상한인 GBDT(Chapter 7)[^gbdt]와 비교해서
스스로 증명해야 한다. 그리고 이 장은 ML1의 마지막 장이다. 다음(ML2)은
정답 `y`가 있는 세계에서 벗어난다 — 정답 대신 **보상**만 주어지고 에이전트가
스스로 데이터를 만들어가는 강화학습·로봇의 세계다[^cs234]. 이번 장은 그 다리를
건너기 전에, 이 학기 전체를 하나의 절차로 닫는 마지막 점검이다.

## 학습 목표

이번 장을 마치면, 다음을 할 수 있다:

- Block B 팀 프로젝트의 4주 파이프라인(문제 정의 → 3분할·전처리·신경망 →
  test 한 번 측정 → 발표)을 수행하고, "왜 고전 ML이 아니라 딥러닝인가"에
  데이터의 구조(정형/비정형)·규모(공유 구조)라는 두 축과 GBDT 비교·학습
  곡선이라는 두 숫자로 답할 수 있다.
- 8.2의 체크리스트에 "왜 딥러닝인가" 항목을 추가해, 다른 팀의 딥러닝
  발표에서 장·절을 명시한 반박 가능한 질문을 쓰고, 그 질문의 품질로
  자신의 peer-review 성적을 만들 수 있다.
- ML1 전체(Chapter 2~15)를 "모델 정의 → 손실로 수치화 → θ 조정 → 과적합
  통제"라는 4단계 구조 위에 올려 공통 흐름을 설명하고, ML2로 넘어갈 때
  무엇이 바뀌는지(정답 `y` → 보상 `r`, 정적 데이터 → 에이전트의 경험) 짚을
  수 있다.

세 블록을 한눈에 보면, Chapter 8과 동일한 분업이다. 16.1은 그 절차를
**코드와 보고서**로 수행하고, 16.2는 **리뷰**로 심사하며, 16.3은
**총정리와 다음 학기 연결**로 마무리한다. 8.1의 마지막 문장 — "같은 절차를
세 가지 언어(코드, 보고서, 리뷰)로 말할 수 있어야 이 프로젝트가 끝난
것" — 이, 이 장 전체의 통과 기준이다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [16.1 팀 프로젝트: 구조와 발표](chapter16/1.md) — Chapter 9~15의 도구를
  실제 데이터에 적용하는 기말 프로젝트의 구조를 정한다. 핵심은 "왜
  딥러닝인가"라는 질문: 무작위·GBDT라는 **두 층의 베이스라인**을 모두
  넘어야 "이 데이터에 딥러닝이 필요했다"고 말할 수 있고, 그 근거를 데이터
  구조·규모라는 두 축과 GBDT 비교·학습 곡선이라는 두 숫자로 반박 가능한
  주장으로 세운다. 4주 파이프라인의 마일스톤(M1~M4), "왜 CNN인가"를
  파라미터 수 공식(10.1)으로 손으로 계산하는 수식적 정당화, 에포크를
  x축으로 하는 학습 곡선을 다룬다.
- [16.2 Peer-Review](chapter16/2.md) — 이번 주엔 네가 심사위원이다. 8.2의
  체크리스트에 "왜 딥러닝인가 답변의 타당성"이라는 5번째 항목이 더해지고,
  심사 대상이 고전 ML에서 CNN·Transformer·VAE로 바뀌면서 같은 질문이 더
  깊어진다 — 아키텍처와 데이터 구조의 귀납적 편향이 일치하는가, 에포크를
  val 곡선으로만 정했는가, 확률의 교정은 믿을 만한가. Block B에서는
  하이퍼파라미터 후보가 훨씬 많고 에포크라는 시간 차원이 생겨 선택 편향
  (`E[max]` 부풀림)도 커지므로, 그 크기를 정량화해 "test 0.95 vs 0.94는
  잡음 수준"이라는 3점 질문을 쓰는 법을 실습으로 체득한다.
- [16.3 ML1 총정리와 ML2로 가는 길](chapter16/3.md) — 새 개념이 하나도
  없는, 순수 결실의 블록이다. 프로젝트 채점의 합산(발표 50%, 보고서 30%,
  리뷰 20%)과 "리뷰 한 통이 숫자를 바꾼" 실제 사례를 결실하고, ML1
  전체(Ch02~15)를 4단계 학습 구조와 개념지도 한 장 위에 올려 장을 넘나드는
  복습 문제 7개로 점검한다. 마지막으로 ML1(정답 `y` 있는 지도·비지도학습)과
  ML2(보상 `r`만 있는 강화학습·로봇)를 나란히 놓고, 4단계 구조는 그대로
  살면서 ①의 모델(정책)과 ④의 데이터(에이전트의 경험)만 바뀐다는 "같은
  구조, 다른 질문"을 짚는다.

[^transformer]: Vaswani, A. et al. (2017). "Attention Is All You Need." arXiv:1706.03762.
[^vae]: Kingma, D. P., Welling, M. (2013). "Auto-Encoding Variational Bayes." arXiv:1312.6114.
[^gan]: Goodfellow, I. J. et al. (2014). "Generative Adversarial Networks." arXiv:1406.2661.
[^cs234]: Stanford CS234: Reinforcement Learning. https://web.stanford.edu/class/cs234/
[^ddpm]: Ho, J., Jain, A., Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." arXiv:2006.11239.
[^gbdt]: Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." Annals of Statistics 29(5), 1189–1232.
[^alexnet]: Krizhevsky, A., Sutskever, I., Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." NeurIPS 2012.
[^gpt3]: Brown, T. B., Mann, B., Ryder, N., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS 2020. arXiv:2005.14165.
[^gmm]: McLachlan, G. J., Krishnan, T. (2008). "The EM Algorithm and Extensions." 2nd ed. Wiley.
