# 데이터 가게 커버 이미지 프롬프트 (Genspark 등 이미지 생성용)

## 쓰는 법

1. 아래 **공통 스타일** 을 각 프롬프트 뒤에 붙여서 요청합니다.
2. 결과 그림을 `notebooks/datashop/assets/covers_ai/<이름>.png` 로 저장합니다 (`.jpg` · `.webp` 도 됨). 이름은 각 항목 제목 옆 괄호 안의 영어입니다.
3. `make_shop.py --pages` 와 `worldcup/update_page.py` 를 다시 돌리면, 그 파일이 있는 데이터는 Wikimedia 사진 대신 생성 그림을 씁니다 (출처: "AI 생성 이미지").

권장 크기: 가로로 긴 **16:5 (예: 1600×500)**. 월드컵 카드에는 가운데 부분이 8:5 로 잘려 들어가니, 중요한 것은 가운데에 둡니다.

## 공통 스타일 (모든 프롬프트 뒤에 붙이기)

```
Wide cinematic banner, 16:5 aspect ratio, photorealistic editorial photography, vivid but natural colors, soft daylight, clean composition, main subject in the center third, no text, no letters, no numbers, no logos, no brand names, no watermarks, no recognizable real people.
```

## 커머스 · 콘텐츠

1. **비슷한 옷 추천** (`fashion_commerce`)
   `A bright modern clothing boutique with colorful shirts and dresses neatly arranged on racks, a few items highlighted as if being recommended, warm store lighting`
2. **영화 추천** (`movie_ratings`)
   `A cozy living room at night with a large TV showing a grid of blurred movie posters, a hand holding a remote, popcorn bowl on a sofa, soft blue screen glow`
3. **쇼핑몰 구매 전환** (`shop_conversion`)
   `A person's hands holding a smartphone showing an online shopping cart screen (blurred, no text), cosmetics products softly blurred on a desk in the background`
4. **한국어 리뷰 감성** (`korean_reviews`)
   `A Korean movie theater lobby at night with colorful posters and neon signs, people writing reviews on their phones, Seoul city atmosphere`
5. **광고 클릭률 (CTR)** (`ad_ctr`)
   `A busy city square at dusk covered with large colorful digital billboards, abstract shapes on the screens, crowds walking by, motion blur`

## 금융 · 핀테크

6. **카드 이상거래 (보이스피싱 · 자금세탁 등)** (`card_fraud`)
   `Close-up of a contactless card payment terminal at a shop counter, a red warning glow on the terminal screen, a shadowy hand in the background, tense mood`
7. **자금세탁 거래 탐지** (`aml_transactions`)
   `Abstract visualization of money flowing between bank icons and people icons as glowing lines in a network, some loops highlighted in red, dark navy background`
8. **주가 지수: 내일 오를까?** (`stock_index`)
   `A trading desk with several monitors showing green and red line charts and candlesticks (no numbers), city skyline through the window at sunrise`
9. **대출 상환 · 신용평가** (`credit_default`)
   `A modern bank branch counter with a loan officer reviewing documents with a customer (faces not visible), bright clean interior, calm professional mood`
10. **법인카드 지출 관리** (`corporate_card`)
    `A business traveler's desk with a corporate credit card, paper receipts, a laptop showing colorful expense category charts, coffee cup, top-down view`

## 플랫폼 · 서비스

11. **고객 문의 자동 분류** (`support_routing`)
    `A bright customer support center with agents wearing headsets in front of screens, chat bubbles sorted into colorful lanes floating above them`
12. **호텔 예약: 취소와 가격** (`hotel_booking`)
    `A sunny seaside resort hotel in Portugal with a pool and white buildings, a laptop on a lounge chair showing a calendar with some dates crossed out`
13. **연결 예측: 친구 추천** (`link_prediction`)
    `Illustration of five diverse friendly people as round avatar icons connected by lines in a social network, two new dashed connections glowing, clean white background`
14. **네트워크 침입 탐지** (`cyber_security`)
    `A security operations center with large screens showing a glowing network map, one red alert node among blue nodes, dark room with blue lighting`
15. **통신사 고객 이탈** (`telco_churn`)
    `A mobile phone store counter with a customer about to leave through the door, phones on display, cool evening light, a subtle sense of departure`

## 도시 · 공간

16. **뉴욕 택시 수요 (시공간)** (`taxi_demand`)
    `Aerial view of Manhattan avenues at dusk filled with yellow taxis, a semi-transparent heat map glowing over the city blocks`
17. **자율주행: 옆 차 궤적 예측** (`vehicle_trajectory`)
    `Bird's-eye view of a multi-lane highway with cars, each car leaving a glowing predicted path line ahead of it, one car's path curving into the next lane`
18. **집값 예측** (`real_estate`)
    `Aerial view of a sunny California suburb with colorful roofs and palm trees near the coast, a soft gradient overlay from cool to warm colors`
19. **지진: 시공간 데이터** (`earthquakes`)
    `A world map seen from above with glowing dots along the Pacific Ring of Fire, ripples spreading from a few points, dark ocean and teal land`
20. **도로 교통 속도 (내비)** (`traffic_speed`)
    `Los Angeles freeway interchange at sunset with streams of car headlights and taillights, road segments colored green, yellow and red like a navigation app`
21. **미세먼지 예보** (`air_quality`)
    `A city skyline split in half: left side clear blue sky, right side hazy orange smog, people wearing masks on the hazy side, wide view`
22. **전력 수요 예측** (`electricity`)
    `An apartment building at night with some windows lit and glowing power lines leading to a substation, a smart meter on the wall in the foreground`

## 제조

23. **반도체 공정 불량** (`semiconductor_yield`)
    `A clean room with a technician in a white suit holding a shiny silicon wafer reflecting rainbow colors, yellow lighting`
24. **설비 고장 예측 (예지보전)** (`predictive_maintenance`)
    `A modern CNC milling machine in a factory with a tablet next to it showing sensor graphs, sparks and metal shavings, industrial blue and orange tones`
25. **강판 결함 분류** (`steel_defects`)
    `A steel rolling mill with a glowing hot steel sheet moving on rollers, an inspection camera above it, orange heat and blue steel tones`
26. **베어링 진동 진단** (`bearing_vibration`)
    `Close-up of a metal ball bearing inside an electric motor with a vibration sensor attached, a glowing waveform line in the air, workshop background`

## 이미지 · 소리 · 3D

27. **3D 포인트 클라우드** (`point_cloud`)
    `A 3D scan made of thousands of colorful glowing points forming a chair, an airplane and a lamp floating in a dark space`
28. **흉부 X-ray 폐렴 찾기** (`medical`)
    `A radiologist's workstation with a chest X-ray on a monitor, a highlighted region on the lung area, calm blue hospital lighting (no faces)`
29. **위성 사진 토지 분류** (`satellite`)
    `A satellite view of European countryside: patchwork of farm fields, a forest, a river, a highway and a small town, crisp colors from orbit`
30. **작물 병해 진단** (`crop`)
    `A farmer's hand holding a smartphone that photographs a tomato leaf with brown spots in a green field, morning light`
31. **음성 명령 인식** (`speech`)
    `A smart speaker on a kitchen table with a colorful glowing sound wave rising from it, a person speaking in the soft-focus background`

## 스포츠 · 게임

32. **축구: 이 슛은 골일까?** (`football`)
    `A football striker taking a shot toward the goal in a packed stadium at night, a glowing arc showing the ball's path, dramatic floodlights`
33. **체스: 누가 이길까?** (`chess`)
    `A wooden chess board mid-game in a park, two players' hands about to move pieces, warm afternoon light, shallow depth of field`

## 전문 서비스

34. **문서 종류 자동 분류** (`document_types`)
    `Scanned paper documents of different kinds (letter, invoice, form, resume, newspaper) flying into colorful labeled folders without readable text, office background`
35. **AI 활용사례 찾기** (`ai_usecases`)
    `A bright modern office where workers at different jobs (designer, programmer, accountant) each have a small glowing AI assistant icon by their screen`
36. **채용: 이력서 · 공고 매칭** (`hiring_match`)
    `Two documents, a resume and a job posting, side by side on a desk with glowing lines connecting matching sections, a handshake softly blurred in the background`
