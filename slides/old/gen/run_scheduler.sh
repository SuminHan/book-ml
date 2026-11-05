#!/bin/bash
# ---------------------------------------------------------------------------
# KSA 슬라이드 일괄 생성 스케줄러
#   - gen/manifest.json 의 pending 덱을 하나씩 로컬 Qwen(cq)에게 시켜 만든다
#   - 각 덱: cq 가 .tex 작성 -> 스케줄러가 그림 변환 + xelatex 2회 + 상태 갱신
#   - 매 덱마다 gen/dashboard.html / gen/STATUS.txt 재생성
# 사용:  nohup bash gen/run_scheduler.sh > gen/logs/scheduler.log 2>&1 &
#   또는 Bash 도구 run_in_background:true 로 실행
# 재시작 안전: 이미 done 인 덱은 건너뛴다.
# ---------------------------------------------------------------------------
set -u
SLIDES="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SLIDES"
GEN="$SLIDES/gen"
MAN="$GEN/manifest.json"
LOCK="$GEN/.scheduler.lock"
PER_DECK_TIMEOUT="${PER_DECK_TIMEOUT:-4200}"     # 70분/덱 상한
MIN_PAGES="${MIN_PAGES:-28}"                      # 3시간 덱 최소 쪽수(휴리스틱)
MAX_DECKS="${MAX_DECKS:-0}"                       # >0 이면 그만큼만 처리(테스트용)
_done_count=0

# --- local Qwen env (mirrors the cq() shell function) ---
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export ANTHROPIC_BASE_URL="http://localhost:4000"
export ANTHROPIC_AUTH_TOKEN="sk-local-litellm-key"
export ANTHROPIC_MODEL="qwen3.8-27b"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="qwen3.8-27b"

if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[sched] already running (pid $(cat "$LOCK"))"; exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

jqset() {  # <id> <key> <json-value>
  python3 - "$MAN" "$1" "$2" "$3" <<'PY'
import json,sys
man,did,key,val=sys.argv[1:5]
d=json.load(open(man,encoding="utf-8"))
try: val=json.loads(val)
except Exception: pass
for x in d["decks"]:
    if x["id"]==did: x[key]=val
json.dump(d,open(man,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
PY
}
dash() { python3 "$GEN/build_dashboard.py" >/dev/null 2>&1 || true; }
now() { date +%s; }

next_pending() {
  python3 - "$MAN" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
for x in d["decks"]:
    if x["status"]=="pending":
        print(x["id"]); break
PY
}

deck_field() {  # <id> <key>
  python3 - "$MAN" "$1" "$2" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
for x in d["decks"]:
    if x["id"]==sys.argv[2]: print(x.get(sys.argv[3],"")); break
PY
}

build_prompt() {  # <id> -> writes gen/prompts/<id>.md, echoes path
  local id="$1" p="$GEN/prompts/$1.md"
  local title course_name week chapter overview
  title=$(deck_field "$id" title)
  course_name=$(deck_field "$id" course_name)
  week=$(deck_field "$id" week)
  chapter=$(deck_field "$id" chapter)
  local secs; secs=$(python3 - "$MAN" "$id" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
for x in d["decks"]:
    if x["id"]==sys.argv[2]:
        print("\n".join("  - "+s for s in x["src_sections"]))
PY
)
  {
    cat "$GEN/deck_instructions.md"
    echo
    echo "---"
    echo "## 이번 작업 (이 덱만)"
    echo
    echo "- 과목: **$course_name**, **${week}주차**, $chapter"
    echo "- 챕터 제목: $title"
    echo "- 개요 파일(먼저 읽기): \`$(deck_field "$id" src_overview)\`"
    echo "- 차시 파일들(각각 1시간 분량, 순서대로 \\section 하나씩):"
    echo "$secs"
    echo "- 스타일 예시(읽기): \`gen/REFERENCE-deck.tex\`, 테마: \`theme/ksa-theme.tex\`"
    echo "- **출력 파일(이것만 Write)**: \`$(deck_field "$id" out_tex)\`"
    echo
    echo "위 파일들을 Read 로 읽고, 지시대로 3시간(50~66프레임, 4차시면 65~85) 분량"
    echo "슬라이드 \`.tex\` 를 만들어 지정 경로에 Write 해라. 그림 변환·컴파일은 하지 마라."
    echo "다른 파일은 건드리지 마라. 끝나면 한 줄로 '완료: <경로> (<프레임수>프레임)' 만 말해라."
  } > "$p"
  echo "$p"
}

echo "[sched] start $(date '+%F %T')  slides=$SLIDES"
python3 "$GEN/init_manifest.py" >/dev/null 2>&1 || true
dash

while :; do
  ID=$(next_pending)
  [ -z "$ID" ] && { echo "[sched] no pending decks left. done."; dash; break; }

  OUT_TEX=$(deck_field "$ID" out_tex)
  ATT=$(deck_field "$ID" attempts); ATT=$((ATT+1))
  T0=$(now)
  echo "[sched] === $ID  (attempt $ATT)  $(date '+%T') ==="
  jqset "$ID" status '"running"'
  jqset "$ID" attempts "$ATT"
  jqset "$ID" started "$T0"
  jqset "$ID" notes '""'
  dash

  PROMPT=$(build_prompt "$ID")
  LOG="$GEN/logs/$ID.log"

  timeout "$PER_DECK_TIMEOUT" claude -p "$(cat "$PROMPT")" \
      --dangerously-skip-permissions --add-dir "$SLIDES/.." \
      > "$LOG" 2>&1
  CQRC=$?
  echo "[sched] cq exit=$CQRC  $ID"

  if [ ! -f "$SLIDES/$OUT_TEX" ]; then
    jqset "$ID" status '"failed"'
    jqset "$ID" notes "\"cq가 $OUT_TEX 를 생성하지 않음 (exit $CQRC). 로그: gen/logs/$ID.log\""
    jqset "$ID" finished "$(now)"
    jqset "$ID" duration_s "$(( $(now) - T0 ))"
    dash; sleep 3; continue
  fi

  # cq가 자주 하는 실수 자동 보정 + 없는 그림 참조 무력화 + 컴파일
  bash "$GEN/fix_tex.sh" "$SLIDES/$OUT_TEX" >> "$LOG" 2>&1
  bash "$SLIDES/build_figs.sh" "$OUT_TEX" >> "$LOG" 2>&1
  ( cd "$SLIDES/kor" && \
    xelatex -interaction=nonstopmode -halt-on-error "$(basename "$OUT_TEX")" \
      >> "$LOG" 2>&1 ; \
    xelatex -interaction=nonstopmode "$(basename "$OUT_TEX")" >> "$LOG" 2>&1 )
  PDF="$SLIDES/kor/$(basename "${OUT_TEX%.tex}").pdf"

  FRAMES=$(grep -cE '\\begin\{frame\}' "$SLIDES/$OUT_TEX" 2>/dev/null || echo 0)
  jqset "$ID" frames "$FRAMES"

  if [ -f "$PDF" ]; then
    PAGES=$(pdfinfo "$PDF" 2>/dev/null | awk '/^Pages:/{print $2}')
    PAGES=${PAGES:-0}
    jqset "$ID" pages "$PAGES"
    ERRS=$(grep -cE '^! ' "$LOG" || true)
    if [ "$PAGES" -ge "$MIN_PAGES" ] && [ "${ERRS:-0}" -le 3 ]; then
      jqset "$ID" status '"done"'
      jqset "$ID" notes "$( [ "${ERRS:-0}" -gt 0 ] && echo "\"경미한 tex오류 ${ERRS}건(자동보정 후 잔여) — 마감 검토 시 확인\"" || echo '""' )"
    else
      jqset "$ID" status '"needs_review"'
      jqset "$ID" notes "\"pdf=$PAGES쪽, tex오류=${ERRS}줄, 프레임=$FRAMES. gen/logs/$ID.log 확인\""
    fi
    git -C "$SLIDES/.." add "slides/$OUT_TEX" "slides/kor/$(basename "$PDF")" 2>/dev/null || true
  else
    jqset "$ID" status '"failed"'
    jqset "$ID" notes "\"xelatex 실패, PDF 없음. gen/logs/$ID.log 확인\""
  fi

  jqset "$ID" finished "$(now)"
  jqset "$ID" duration_s "$(( $(now) - T0 ))"
  dash
  ST=$(deck_field "$ID" status)
  bash "$GEN/publish_and_push.sh" "slides($ID): $ST ${FRAMES}f — $(deck_field "$ID" title | cut -c1-50)" >> "$GEN/logs/push.log" 2>&1 || true
  echo "[sched] $ID -> $ST  ${FRAMES}f  $(( ($(now)-T0)/60 ))m"
  _done_count=$((_done_count+1))
  if [ "$MAX_DECKS" -gt 0 ] && [ "$_done_count" -ge "$MAX_DECKS" ]; then
    echo "[sched] MAX_DECKS=$MAX_DECKS reached, stopping."; break
  fi
  sleep 5
done

echo "[sched] end $(date '+%F %T')"
touch "$GEN/.scheduler.done"
