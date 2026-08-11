# Problem Set

**1.** (코딩) 다음과 같은 함수 `simple_tokenize`와 `next_token_distribution`을
작성하라 — 아주 단순화된 언어모델 시뮬레이션이다.

- `simple_tokenize(text)`: 공백 기준으로 텍스트를 토큰(단어) 리스트로 나눈다.
- `next_token_distribution(corpus_tokens, context_word)`: 학습 말뭉치(토큰 리스트)에서
  `context_word` 바로 다음에 등장한 단어들의 빈도로 확률분포(딕셔너리)를 만든다.

```python
def simple_tokenize(text):
    # ADD ADDITIONAL CODE HERE!!

def next_token_distribution(corpus_tokens, context_word):
    # ADD ADDITIONAL CODE HERE!!
    # context_word가 등장한 모든 위치에서, 바로 다음 토큰을 세어
    # {다음단어: 확률} 딕셔너리로 반환

corpus = simple_tokenize("나는 학교에 간다 나는 집에 간다 나는 학교에 있다")
print(next_token_distribution(corpus, "나는"))
# {"학교에": 2/3, "집에": 1/3}
print(next_token_distribution(corpus, "학교에"))
# {"간다": 1/2, "있다": 1/2}
```

**2.** (코딩) 문제 1의 `next_token_distribution`을 이용해, 주어진 시작 단어에서
매번 **가장 확률이 높은** 다음 단어를 골라 이어붙이는 `greedy_generate(corpus_tokens,
start_word, length)`를 작성하라 (동률이면 사전순으로 먼저 오는 단어를 선택).

```python
def greedy_generate(corpus_tokens, start_word, length):
    # ADD ADDITIONAL CODE HERE!!

print(greedy_generate(corpus, "나는", 3))  # ["나는", "학교에", "간다"]
```

**3.** (프롬프트 설계, 코딩 아님) 다음 두 프롬프트 중 어느 쪽이 chain-of-thought
기법을 쓴 것인지 표시하고, 왜 복잡한 산술/논리 문제에서 chain-of-thought 프롬프트가
더 정확한 답을 내는 경향이 있는지 한 문단으로 설명하라.

> (A) "23×17은 얼마인가? 답만 말해줘."
>
> (B) "23×17을 계산해줘. 단계별로 어떻게 계산하는지 먼저 보여준 다음 최종 답을
> 말해줘."

**4.** Few-shot 프롬프팅과 파인튜닝은 둘 다 "모델이 새로운 형식의 작업을 하게
만든다"는 점에서 비슷해 보인다. 두 방법의 차이를 (a) 모델의 가중치가 바뀌는지 여부,
(b) 필요한 데이터/계산 비용의 관점에서 비교하라.
