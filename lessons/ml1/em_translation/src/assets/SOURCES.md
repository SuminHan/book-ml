# 출처
- v3 말뭉치(예시, 어순이 꼬인 5쌍): ich habe das Buch gelesen / ich habe das Haus gesehen / heute lese ich das Buch / heute sehe ich das Haus / morgen lese ich. data.py assert: 5단어 문장 짝 1/5, das–the 1/5×4=0.8개, das 전체 4개 → M1 das→the 20% = das→I 20% 동점, ich→I 23% 단독 1등, 50번 11단어 모두 정답 93~100%(das→the 97%, das→I 3%). 실패 예: das Haus ist klein + klein ist das Buch 는 ist·klein 이 늘 함께 나와 50:50에서 멈춤
- v2 소스: data_v2.py · shorts/lines_v2.json · shorts_v2.py, 영상·캡션·표지 _v2
- v2 말뭉치(예시): Koehn 예제를 늘린 5쌍 — das Haus ist klein / das Buch ist neu / ich lese das Buch / ich sehe das Haus / heute lese ich (어순 바뀜). data.py assert: 처음 4단어 문장 짝 1/4, das–the 1/4×4=1개, das 전체 4개 → M1 das→the 25%(나머지 ≤12.5%), 30번 10단어 모두 정답 96~100%. 수식 SVG: src/make_figs.py (matplotlib mathtext)
- v1(3쌍) 소스: data_v1.py · shorts/lines_v1.json · shorts_v1.py, 영상·캡션·표지 _v1
- v1 예제 말뭉치: Philipp Koehn, *Statistical Machine Translation* (Cambridge University Press, 2010), 4.2절 IBM Model 1 EM 예제 (das Haus / the house, das Buch / the book, ein Buch / a book)
- 방법: P. F. Brown, S. A. Della Pietra, V. J. Della Pietra, R. L. Mercer, "The Mathematics of Statistical Machine Translation: Parameter Estimation", *Computational Linguistics* 19(2), 1993 (IBM 연구진, Model 1–5 를 EM 으로 학습)
- EM 일반: week04 p60·p65–p66 (E 단계·M 단계), Dempster·Laird·Rubin 1977
- 계산: src/data.py (t 를 1/4 로 시작, 20번 반복) assert: 처음 모든 짝 0.5, das–the 기대 개수 1, M 1번 das→the 50%·Haus the/house 50:50, 2번째 E Haus–house 67%, 20번 das→the·Buch→book 100%, Haus→house·ein→a 97%
- 이모지: Twemoji (jdecked/twemoji, 그래픽 CC BY 4.0)
