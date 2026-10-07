<!-- make_readme.py 로 자동 생성: 고치려면 lines.json · 캡션 · SOURCES.md 를 고치고 다시 실행 -->
# 결정 트리

*질문으로 데이터를 나누는 분류기*

> 결정 트리(Decision Tree)로 붓꽃(Iris) 3종 분류하기

## 영상 내용

- 스무고개처럼 예·아니오 질문을 이어 가며 데이터를 나누는 방법이에요. 붓꽃 150송이를 꽃잎 길이·너비만으로 세 종(setosa, versicolor, virginica)으로 갈라 봤어요.

- • 좋은 질문 = 나눈 뒤 한쪽이 깨끗해지는 질문. 섞인 정도(불순도)를 가장 많이 줄이는 질문부터 골라요. 꽃잎 길이가 2.5cm보다 짧으면 setosa 50송이가 전부 갈려요.
- • 남은 칸에서도 반복해요. 꽃잎 너비가 1.75cm보다 좁으면 versicolor(49송이 + 섞인 5송이) / virginica(45송이 + 섞인 1송이)로 나뉘고, 한 종이 대부분 남으면 멈춰요. 이 끝을 잎(leaf)이라고 해요.
- • 새 꽃은 질문을 따라 내려가면 종류가 나와요.
- • 너무 깊게 키우면 경계에 걸친 몇 송이(6송이)까지 맞추려고 칸이 잘게 쪼개져요(잎 3개 → 8개, 깊이 2 → 5). 새 데이터에는 오히려 틀리기 쉬워서(과적합) 깊이를 제한해요.

- 데이터: Iris (R. A. Fisher, 1936)
- 사진: Radomil (setosa) · Dlanglois (versicolor) — CC BY-SA 3.0 / Frank Mayfield (virginica) — CC BY-SA 2.0, Wikimedia Commons

## 장면별 자막과 대사

| # | 화면 자막 | 대사 |
|---|---|---|
| 1 | 붓꽃(iris) 3종을 꽃잎 크기만 보고 / 질문 몇 개로 가를 수 있을까? | 붓꽃 백오십 송이를, 꽃잎 길이와 너비만 보고 질문 몇 개로 세 종류로 가를 수 있을까요? |
| 2 | 질문마다 나뉘는 정도가 다르다 / 한쪽이 깨끗해지는 질문이 좋다 | 질문마다 효과가 달라요. 이 질문은 양쪽이 여전히 섞여 있고, 이 질문은 한쪽이 세토사만 남아요. |
| 3 | 섞인 정도(불순도)를 / 가장 많이 줄이는 질문부터 | 그래서 섞인 정도가 가장 많이 줄어드는 질문을 골라요. 꽃잎이 짧으면 무조건 세토사예요. |
| 4 | 남은 칸에서 / 같은 방식으로 반복 | 남은 칸에서도 같은 일을 반복해요. 이번엔 꽃잎 너비가 1.75보다 작은지 물어요. |
| 5 | 한 종이 대부분 남으면 멈춘다 / 끝 = 잎 (leaf) | 한 종이 대부분 남으면 멈추고, 이 끝을 잎이라고 불러요. |
| 6 | 새 꽃은 질문을 따라 내려가면 / 종류가 나온다 | 새 꽃은 질문을 따라 내려가면, 종류가 바로 나와요. |
| 7 | 너무 깊으면 경계의 몇 송이까지 외운다 / = 과적합 → 깊이 제한 | 하지만 끝까지 키우면, 경계에 걸친 몇 송이까지 맞추려고 잘게 쪼개요. 이걸 과적합이라 하고, 그래서 깊이를 제한해요. |
| 8 | 질문으로 나누기 · 좋은 질문 고르기 / 깊이는 적당히 | 정리하면, 결정 트리는 질문으로 나누고, 섞인 정도를 가장 줄이는 질문을 고르고, 너무 깊지 않게 키워요. |

## 출처

파일 | 원본 | 작성자 | 라이선스
iris_setosa.jpg | https://commons.wikimedia.org/wiki/File:Kosaciec_szczecinkowaty_Iris_setosa.jpg | Radomil | CC BY-SA 3.0 (GFDL 재라이선스)
iris_versicolor.jpg | https://commons.wikimedia.org/wiki/File:Iris_versicolor_3.jpg | Dlanglois | CC BY-SA 3.0
iris_virginica.jpg | https://commons.wikimedia.org/wiki/File:Iris_virginica.jpg | Frank Mayfield | CC BY-SA 2.0
crop_*.jpg 는 위 사진을 3:4로 자른 것 (make_figs.py)

## 파일

- 영상: `~/book-ml/videos/주제별/결정트리/` (영상·제작 원본은 git 에 올리지 않음)
- 다시 만들기: `~/Documents/manim_shorts/_env/v/bin/python src/make.py shorts`
