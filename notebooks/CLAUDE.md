# notebooks/ — Colab 실습 노트북 작업 규칙

학생은 GitHub main 브랜치의 파일을 Colab으로 연다
(`https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/<ml1|ml2>/<file>.ipynb`).
**push 하기 전까지는 Colab에 반영되지 않는다.**

## 도구: `nb` (raw JSON 편집 금지)
```
nb ls   F                  셀 목록 (번호는 0부터)
nb cat  F 5   | nb cat F 3-7   셀 내용 보기
nb set  F 5 <<'EOF'        셀 5 내용을 통째로 교체 (코드셀이면 옛 출력도 지워짐)
...새 내용...
EOF
nb add  F 6 code <<'EOF'   6번 자리에 새 셀 삽입 (md 셀은 code 대신 md, 맨 끝은 end)
nb rm   F 5                셀 삭제
nb mv   F 5 2              셀 이동
nb check F                 JSON/Colab 배지/문법 검사 (빠름)
nb run  F                  전체 실행 검증 (파일은 안 바뀜)
nb run  F --save           실행하고 출력까지 노트북에 저장
nb new  F "제목"           배지+설정셀이 든 새 노트북
```
- `nb set`은 셀 **전체**를 바꾼다. 일부만 고칠 때도 `nb cat`으로 읽은 원문을 바탕으로 전체를 다시 쓴다.
- 셀 번호는 add/rm 후 바뀐다. 다시 `nb ls` 하고 진행.
- `nb run`은 전용 환경(numpy, matplotlib, sklearn, scipy, pandas, torch, gymnasium)에서 돈다. `!pip`, `%` 매직 줄은 건너뛴다.

## 작업 순서
1. `nb ls F`로 구조 파악 → 고칠 셀만 `nb cat`.
2. `nb set/add/rm`으로 수정.
3. `nb check F` → `nb run F`. 실패하면 "FAILED at cell [N]"의 셀을 고치고 다시.
4. 통과하면 `nb run F --save`로 출력 갱신(기존 노트북은 대부분 출력이 저장된 상태).
5. `git diff --stat`으로 바뀐 파일 확인. 커밋/푸시는 사용자가 시킬 때만.

## 노트북 내용 규칙
- 첫 셀: `# 제목` + Open in Colab 배지(링크 경로 = 실제 파일 경로). 파일 이름을 바꾸면 배지 링크도 고친다(`nb check`가 잡아준다).
- 두 번째·세 번째 셀: `## 0. 설정: 한글 폰트와 import` + 폰트 설정 코드셀. 지우지 않는다.
- 설명(markdown)은 한국어, 코드 주석도 한국어. 수식은 `\( ... \)`.
- numpy/matplotlib 위주. 프레임워크가 꼭 필요할 때만 torch.
- 책 본문과 숫자가 맞는지 `assert`로 확인하는 셀이 있으면 지우지 말 것.
- 일부 노트북은 그림을 `../../kor/src/images`(책 그림)에 저장한다. `nb run`은 임시 폴더에서 돌아서 책 그림을 건드리지 않는다. `--here`는 사용자가 "책 그림도 다시 뽑아"라고 할 때만 쓴다.
- 파일명: `chapterNN_K_주제.ipynb`. 새 노트북을 만들면 `notebooks/ml1/README.md` 표에도 한 줄 추가.
