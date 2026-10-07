<!-- make_readme.py 로 자동 생성: 고치려면 lines.json · 캡션 · SOURCES.md 를 고치고 다시 실행 -->
# EM 알고리즘 번역

*EM 알고리즘 · IBM Model 1 · 단어 정렬*

> EM 알고리즘(Expectation-Maximization)으로 사전 없이, 어순이 꼬인 번역 문장 5쌍만 가지고 독일어 단어 11개의 뜻(번역 짝)을 배우는 과정이에요. 1993년 IBM 통계 번역 모델(IBM Model 1)의 방식이에요.

## 영상 내용

- 번역된 문장 5쌍. 어느 단어끼리 짝인지 모르고 어순도 꼬여 있어요: ich habe das Buch gelesen = I have read the book (동사가 맨 뒤), heute lese ich das Buch = today I read the book (동사가 주어 앞)
- 처음엔 모든 짝을 똑같이 믿어요 (사전: 모든 뜻 1/10)
- E 단계(Expectation, 기댓값): 짝일 확률만큼 나눠 세요. das–the는 네 문장에서 1/5씩, 합쳐서 0.8개
- 식: δ(f,e) = t(e|f) ÷ (문장 속 모든 독일어 후보의 t 합)
- M 단계(Maximization, 최대화): 센 개수로 사전을 다시 써요. das가 센 짝은 모두 4개, the는 0.8개 → 20%. 그런데 I도 20%로 동점
- 식: t(e|f) = count(f,e) ÷ (f가 센 모든 개수의 합)
- 반복하면 ich가 I를 가져가서 das에겐 the만 남아요 (50번: the 97%, I 3%). 꼬인 선도 풀려 11단어 모두 정답 짝 (93~100%)

**참고**: 예시 문장(Koehn, Statistical Machine Translation 2010 의 das Haus/das Buch 예제를 늘림) · 방법: Brown et al., The Mathematics of Statistical Machine Translation (1993). 수치는 직접 계산했어요.

**참고**: 이모지: Twemoji (CC BY 4.0)

## 장면별 자막과 대사

| # | 화면 자막 | 대사 |
|---|---|---|
| 1 | 번역된 문장 5쌍만으로 / 단어 뜻을 알아낼 수 있을까? | 번역된 문장 다섯 쌍만으로 단어 뜻을 알아낼 수 있을까요? |
| 2 | 짝도 모르고, 어순도 꼬여 있다 / 독일어는 동사가 맨 뒤로 가기도 | 짝도 모르고, 어순도 꼬여 있어요. 동사가 맨 뒤로 가기도 해요. |
| 3 | EM 알고리즘 / 처음엔 모든 짝을 똑같이 (1/10) | 이엠 알고리즘. 처음엔 모든 뜻을 10분의 1로 똑같이 믿어요. |
| 4 | E 단계: 확률만큼 나눠 세기 / das–the: 1/5 × 4문장 = 0.8개 | 기댓값 단계. 짝일 확률만큼 나눠 세요. 다스와 더는 네 문장에서 5분의 1씩, 합쳐서 0.8개. |
| 5 | E 단계 식 / 이 짝의 확률 ÷ 문장 속 후보 합 | 식으로는, 이 짝의 확률을 문장 속 후보들의 확률 합으로 나눠요. |
| 6 | M 단계: 센 개수로 사전 다시 쓰기 / das → the 20%, I 20% 동점 | 최대화 단계. 다스가 센 개수는 4개, 그중 더는 0.8개라 20퍼센트. 그런데 아이도 동점. |
| 7 | 반복하면 ich가 I를 가져가고 / das에겐 the만 남는다 | 반복하면 이히가 아이를 가져가서, 다스에겐 더만 남고 꼬인 선도 풀려요. |
| 8 | 50번 반복: 11단어 모두 정답 / 꼬인 어순 그대로 | 50번 만에 열한 단어 모두 짝을 찾았어요. 꼬인 어순 그대로. |
| 9 | 나눠 세고, 사전 고치고, 반복 / 1993년 IBM 번역 모델의 방법 | 확률로 나눠 세고, 개수로 사전을 고치고, 반복하기. 1993년 IBM 번역 모델의 방법이에요. |

## 출처

- v3 말뭉치(예시, 어순이 꼬인 5쌍): ich habe das Buch gelesen / ich habe das Haus gesehen / heute lese ich das Buch / heute sehe ich das Haus / morgen lese ich. data.py assert: 5단어 문장 짝 1/5, das–the 1/5×4=0.8개, das 전체 4개 → M1 das→the 20% = das→I 20% 동점, ich→I 23% 단독 1등, 50번 11단어 모두 정답 93~100%(das→the 97%, das→I 3%). 실패 예: das Haus ist klein + klein ist das Buch 는 ist·klein 이 늘 함께 나와 50:50에서 멈춤
- v2 소스: data_v2.py · shorts/lines_v2.json · shorts_v2.py, 영상·캡션·표지 _v2
- v2 말뭉치(예시): Koehn 예제를 늘린 5쌍 — das Haus ist klein / das Buch ist neu / ich lese das Buch / ich sehe das Haus / heute lese ich (어순 바뀜). data.py assert: 처음 4단어 문장 짝 1/4, das–the 1/4×4=1개, das 전체 4개 → M1 das→the 25%(나머지 ≤12.5%), 30번 10단어 모두 정답 96~100%. 수식 SVG: src/make_figs.py (matplotlib mathtext)
- v1(3쌍) 소스: data_v1.py · shorts/lines_v1.json · shorts_v1.py, 영상·캡션·표지 _v1
- v1 예제 말뭉치: Philipp Koehn, *Statistical Machine Translation* (Cambridge University Press, 2010), 4.2절 IBM Model 1 EM 예제 (das Haus / the house, das Buch / the book, ein Buch / a book)
- 방법: P. F. Brown, S. A. Della Pietra, V. J. Della Pietra, R. L. Mercer, "The Mathematics of Statistical Machine Translation: Parameter Estimation", *Computational Linguistics* 19(2), 1993 (IBM 연구진, Model 1–5 를 EM 으로 학습)
- EM 일반: week04 p60·p65–p66 (E 단계·M 단계), Dempster·Laird·Rubin 1977
- 계산: src/data.py (t 를 1/4 로 시작, 20번 반복) assert: 처음 모든 짝 0.5, das–the 기대 개수 1, M 1번 das→the 50%·Haus the/house 50:50, 2번째 E Haus–house 67%, 20번 das→the·Buch→book 100%, Haus→house·ein→a 97%
- 이모지: Twemoji (jdecked/twemoji, 그래픽 CC BY 4.0)

## 파일

- 영상: `~/book-ml/videos/주제별/EM알고리즘번역/` (영상·제작 원본은 git 에 올리지 않음)
- 다시 만들기: `~/Documents/manim_shorts/_env/v/bin/python src/make.py shorts`
