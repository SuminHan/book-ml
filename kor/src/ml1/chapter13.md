# Chapter 13. LLM: 사전학습, 프롬프팅, 정렬 (LLM: Pretraining, Prompting & Alignment)

의대생은 처음부터 특정 질병 하나만 배우지 않는다 — 먼저 해부학, 생리학,
약리학 같은 방대한 기초를 몇 년에 걸쳐 익힌 뒤에야, 특정 전공(예:
심장내과)을 짧은 기간 집중적으로 훈련받는다. 대규모 언어모델(LLM)이
학습되는 방식도 놀랍도록 비슷하다: 먼저 인터넷의 방대한 텍스트로 "다음
단어를 예측하는" 아주 단순한 과제를 통해 언어와 세상에 대한 폭넓은 지식을
흡수하고(**사전학습, pretraining**)[^gpt3], 그 다음 특정 목적(대화, 코드 작성,
요약)에 맞춰 비교적 적은 데이터로 다듬는다(**파인튜닝, fine-tuning**). 이
장의 후반부는 그 다듬는 과정을 "사람이 실제로 원하는 방향"으로 더 정교하게
만드는 **포스트트레이닝**(post-training)까지 다룬다. 이 장에서는
Chapter 12의 트랜스포머를 "구조"로가 아니라 **동작하는 도구**로 다시
볼 것이다 — 같은 네트워크가 이번엔 어떻게 학습되고, 어떻게 쓰이고,
어떻게 다듬어지는지를 전 생애 주기(학습 → 사용 → 정렬)로 쫓아가는
여행이기 때문이다.

이 장의 출발점은 Chapter 12에서 배운 어텐션과 트랜스포머[^transformer]다. 자아
어텐션과 위치 인코딩이 당시에는 "시퀀스를 잘 표현하는 장치"였다면,
이제 미래 토큰을 보지 못하게 마스킹한 decoder-only 트랜스포머는
"다음 토큰의 확률분포"를 계산하는 실제 엔진이 된다. 반면 다음 장
(Chapter 14, 표현학습)에서는 이 장의 반대편 끝, 즉 LLM이 의미를
고차원 벡터에 어떻게 압축하는지를 정면에서 다룬다 — PCA,
word2vec[^word2vec], Node2Vec[^node2vec], PageRank는 모두 "단어·노드·데이터를
벡터로 표현한다"는
질문의 변주이며, 이 장에서 자연스럽게 마주하게 될 임베딩과 같은
뿌리에서 자라났다.

## 학습 목표

이 장을 마치면 다음을 할 수 있다:

- 다음 토큰 예측을 "사실상 문장 전체 분포를 학습하는 일"과 동일시할
  수 있고, softmax·교차 엔트로피 손실·퍼플렉시티를 작은 예제에서
  직접 계산할 수 있다.
- BPE[^bpe]가 왜 "단어"가 아니라 "단어 조각"의 어휘를 만들어야 하는지
  설명할 수 있고, 사전학습과 파인튜닝이 데이터·목표·비용에서 무엇이
  다른지 짚을 수 있다.
- zero-shot, few-shot, chain-of-thought[^cot] 프롬프팅이 "프롬프트에 얼마나
  보여줄 것인가"의 스펙트럼 위 어디에 놓이는지 비교할 수 있고,
  temperature·top-k·top-p[^topp]로 생성 행동을 조절할 수 있다.
- SFT, RLHF[^rlhf], DPO[^dpo] 세 경로를 그림으로 구분할 수 있고, "그럴듯한
  텍스트"가 "사람이 원하는 답변"과 왜 다른지(데이터 분포와 인간
  선호의 차이) 설명할 수 있다.

## 세 개의 수업 블록

이번 주는 세 개의 수업 블록으로 진행된다:

- [13.1 언어모델링: 다음 토큰 예측과 사전학습·파인튜닝](chapter13/1.md)
  언어모델을 "다음 토큰을 예측하는 함수"로 정의하고, 확률의 사슬
  분해가 이 정의를 가능하게 하는 이유를 본다. 손으로 다음 토큰 분포를
  직접 세어 보고, BPE 토큰화와 softmax·온도·교차 엔트로피 손실·
  퍼플렉시티를 작은 예제로 확인한 뒤, n-gram[^gpt3]에서 신경망까지의 역사를
  거쳐 "인터넷 텍스트로 사전학습하고 목적별로 파인튜닝한다"는 현대
  LLM의 학습 도식으로 마무리한다.
- [13.2 프롬프팅](chapter13/2.md)
  가중치를 전혀 건드리지 않고 입력(지시문·예시)만으로 모델을 다루는
  방법론이다. zero-shot → few-shot → chain-of-thought의 스펙트럼을
  살펴본 뒤, 왜 "예시 몇 개"가 새 작업을 가르치는지, CoT가 추론을
  실제로 개선하는 메커니즘, temperature·top-k·top-p로 출력의
  "확신도"를 조절하는 방법, 그리고 환각(hallucination)이 생기는
  원인까지 함께 다룬다.
- [13.3 포스트트레이닝: SFT, RLHF, DPO](chapter13/3.md)
  InstructGPT[^instructgpt]의 관찰(작지만 정렬된 모델이 거대한 사전학습
  모델보다 사람에게 더 선호됨)에서 출발해, SFT(이상적인 답변 모방),
  RLHF (비교 데이터로 학습한 Bradley-Terry 보상모델 + PPO[^ppo] 강화학습),
  DPO (보상모델을 통하지 않고 선호 데이터로 정책을 직접 학습)를
  순서대로 짚고, LoRA[^lora]/PEFT와 RAG[^rag]·에이전트[^react]까지 한 장에
  모은 전체 지도를 제시한다[^peft].

![서로게이트 함수 L_CLIP의 한 항(단일 timestep)을 확률비 r의 함수로 그린 그래프 — 왼쪽은 이익이 양수(A>0), 오른쪽은 음수(A<0)인 경우. (원 논문 Figure 1)](../images/ref_ppo.png)

이 장의 주제(언어모델 사전학습, 프롬프팅, 정렬)를 더 깊이 다루는 자료:
[^cs224n]

[^instructgpt]: Ouyang, L. et al. (2022). "Training language models to follow instructions with human feedback." arXiv:2203.02155.
[^transformer]: Vaswani, A. et al. (2017). "Attention Is All You Need." arXiv:1706.03762.
[^word2vec]: Mikolov, T., Chen, K., Corrado, G., Dean, J. (2013). "Efficient Estimation of Word Representations in Vector Space." arXiv:1301.3781.
[^cs224n]: Stanford CS224N: Natural Language Processing with Deep Learning. https://web.stanford.edu/class/cs224n/
[^ppo]: Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
[^rlhf]: Christiano, P. et al. (2017). "Deep reinforcement learning from human preferences." NeurIPS 2017. arXiv:1706.03741.
[^dpo]: Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., Finn, C. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023. arXiv:2305.18290.
[^bpe]: Sennrich, R., Haddow, B., Birch, A. (2016). "Neural Machine Translation of Rare Words with Subword Units." ACL 2016. arXiv:1508.07909.
[^cot]: Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., Zhou, D. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022. arXiv:2201.11903.
[^lora]: Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. arXiv:2106.09685.
[^gpt3]: Brown, T. B., Mann, B., Ryder, N., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS 2020. arXiv:2005.14165.
[^rag]: Lewis, P., Perez, E., Piktus, A., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020. arXiv:2005.11401.
[^peft]: Lialin, V., Deshpande, V., Yao, X., Rumshisky, A. (2023). "Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning." arXiv:2303.15647. (LoRA·prefix·adapter 등 PEFT 기법군을 정리한 참고자료)
[^topp]: Holtzman, A., Buys, J., Du, L., Forbes, M., Choi, Y. (2019). "The Curious Case of Neural Text Degeneration." NAACL 2019. arXiv:1904.09751. (top-p, 즉 nucleus 샘플링을 제안한 논문)
[^node2vec]: Grover, A., Leskovec, J. (2016). "node2vec: Scalable Feature Learning for Networks." KDD 2016. arXiv:1607.00653.
[^react]: Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023. arXiv:2210.03629. — 추론(reasoning) 텍스트와 도구 호출(acting)을 대안적으로 생성하게 해서 LLM이 외부 도구를 스스로 쓰는 에이전트 방식을 정립한 원 논문.
