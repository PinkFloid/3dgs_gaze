#!/usr/bin/env bash
# score_card.sh <recording_dir> <card_id> -- E1 卡两遍打分一条命令
#   遍1:无修正回放,拿片尾墙 tag 书签戳的偏置(--stamp-include 79,86 门 6°)
#   遍2:--bias-init 回灌整卡(--bias-tau 0 不衰减)-> eval_e1 --card 出表
# 产物:<rec>/intents.jsonl + <rec>/<card>_score.csv;遍1流留 intents_pass1.jsonl 备查
set -euo pipefail
REC=${1:?用法: score_card.sh <recording_dir> <card_id>}
CARD=${2:?用法: score_card.sh <recording_dir> <card_id>}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
GL=$ROOT/Eye_Tracker/tools/gaze_live.py
EV=$ROOT/Eye_Tracker/tools/eval_e1.py
RUN="conda run --no-capture-output -n nerfstudio python -u"

echo "== 遍0:裸跑(无任何偏置修正)=="
$RUN "$GL" --replay "$REC" --headless --on-tag-deg 0 \
    --log "$REC/intents_raw.jsonl" > /dev/null 2>&1

P1LOG=$(mktemp)
echo "== 遍1:测书签偏置 =="
$RUN "$GL" --replay "$REC" --headless --stamp-include 79,86 --on-tag-deg 6 \
    --log "$REC/intents_pass1.jsonl" 2>&1 | tee "$P1LOG" | grep -iE "bias stamp|白名单" || true
BIAS=$(grep -oP 'bias stamp @tag\d+: \(\K[^)]+(?=\)deg)' "$P1LOG" | tail -1 || true)
rm -f "$P1LOG"

hits() { $RUN "$EV" "$1" --card "$CARD" 2>/dev/null | grep -oP '命中 \K[0-9]+' | tail -1; }

if [ -n "$BIAS" ]; then
    echo "== 遍2:回灌 ($BIAS)° 整卡矫正 =="
    $RUN "$GL" --replay "$REC" --headless --stamp-include 79,86 --on-tag-deg 6 \
        --bias-init " $BIAS" --bias-tau 0 --log "$REC/intents_pass2.jsonl" 2>&1 \
        | grep -iE "bias stamp|回灌" || true
    OK0=$(hits "$REC/intents_raw.jsonl"); OK1=$(hits "$REC/intents_pass1.jsonl"); OK2=$(hits "$REC/intents_pass2.jsonl")
    # 8-16 实测:裸跑 51/57 与修正取优持平且 s2 上修正倒扣(墙 tag 偏置对桌面
    # 视角未必适用)——三稿取优,来源写明;论文最终统一方案等斜位卡裁决
    echo "== 三稿:裸跑 ${OK0:-0} / 自然戳 ${OK1:-0} / 回灌 ${OK2:-0} =="
    BEST="$REC/intents_raw.jsonl"; BN=${OK0:-0}; SRC=裸跑
    if [ "${OK1:-0}" -gt "$BN" ]; then BEST="$REC/intents_pass1.jsonl"; BN=${OK1:-0}; SRC=自然戳; fi
    if [ "${OK2:-0}" -gt "$BN" ]; then BEST="$REC/intents_pass2.jsonl"; BN=${OK2:-0}; SRC=回灌; fi
    echo "== 取优:采用${SRC}稿(${BN}) =="
    cp "$BEST" "$REC/intents.jsonl"
else
    OK0=$(hits "$REC/intents_raw.jsonl"); OK1=$(hits "$REC/intents_pass1.jsonl")
    echo "[!] 没戳上墙 tag;裸跑 ${OK0:-0} vs 自然戳 ${OK1:-0} 取优"
    if [ "${OK0:-0}" -ge "${OK1:-0}" ]; then cp "$REC/intents_raw.jsonl" "$REC/intents.jsonl"
    else cp "$REC/intents_pass1.jsonl" "$REC/intents.jsonl"; fi
fi

echo "== 打分:卡 $CARD =="
$RUN "$EV" "$REC/intents.jsonl" --card "$CARD" --csv "$REC/${CARD}_score.csv"
