# lessons/ — 주제별 수업·영상 자료

주차 번호 없이 **주제 단위**로 만든다 (수업계획 순서가 자주 바뀌므로).

공개 저장소에는 **주제 폴더 바로 아래의 md 만** 올린다(README.md 영상 요약 · 수업설계 · 발표자노트). 영상·코드·그림·src/ 는 로컬에만.

```
lessons/
├── _tools/              공통 코드: 슬라이드 · 실습 노트북 · 음성 · 리허설 · Manim 베이스 · 영상 믹스
└── ml1/<주제>/
    ├── README.md        영상 요약 — 캡션·장면별 자막과 대사·출처 (자동 생성: python3 lessons/_tools/make_readme.py)
    ├── 수업설계.md · 발표자노트.md · 슬라이드.pdf
    ├── 실습/학생용.ipynb · 정답.ipynb
    ├── src/             원본 (여기만 고친다)
    │   ├── lesson.py      슬라이드(화면·노트) + 실습 셀 + 영상 설정
    │   ├── data.py · make_figs.py · figs/
    │   ├── concept/       개념 영상 (가로 16:9): lines.json + concept.py
    │   └── shorts/        쇼츠 (세로 9:16): lines.json + shorts.py
    └── build/           중간 파일 (지워도 다시 생성됨, git 제외)

~/book-ml/videos/주제별/<주제>/   ← 완성 영상은 전부 여기 (업로드 묶음은 videos/업로드용/)
    <주제>_쇼츠.mp4 · _쇼츠_캡션.txt   (여성 음성)
    <주제>_개념영상.mp4                 (여성 음성)
    <주제>_리허설.mp4                   (남성 음성, 슬라이드 + 노트)
```

## 새 주제 만들기
1. 기존 주제 폴더의 `src/` 를 복사 → `lesson.py`, `data.py`, `make_figs.py`, `concept/`, `shorts/` 내용 교체
2. `~/Documents/manim_shorts/_env/v/bin/python src/make.py` (전부) 또는 단계 지정:
   `src/make.py figs slides lab rehearsal concept shorts`
3. 다른 음성이 필요하면 `--voice 남,여`

## 원칙
- 영상 하나하나가 독립 쇼츠처럼 완결 ("다음 시간", "실습해 봅시다" 같은 수업 연결 멘트 X)
- 쇼츠: 첫 프레임에 제목, 인스타 안전영역, 60초 이내, 캡션 해시태그 5개 이하, "저장·복습" 멘트 X
- 비유는 하나로 통일 (나이브 베이즈 = 스팸 메일). 어려운 개념은 **직접 세기 → 공식** 순서로
- 슬라이드·영상·실습은 같은 `data.py` 숫자를 쓴다
