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
    OK1=$(hits "$REC/intents_pass1.jsonl"); OK2=$(hits "$REC/intents_pass2.jsonl")
    # 实测两方案各有胜负(偏置整场小幅漂,常数回灌不通吃):双稿取优,来源写明
    if [ "${OK2:-0}" -gt "${OK1:-0}" ]; then
        echo "== 取优:回灌稿 ${OK2} > 自然戳稿 ${OK1} -> 采用回灌 =="
        cp "$REC/intents_pass2.jsonl" "$REC/intents.jsonl"
    else
        echo "== 取优:自然戳稿 ${OK1} >= 回灌稿 ${OK2} -> 采用自然戳 =="
        cp "$REC/intents_pass1.jsonl" "$REC/intents.jsonl"
    fi
else
    echo "[!] 遍1没戳上任何墙 tag(开头结尾都没盯够?)——无修正直接打分"
    cp "$REC/intents_pass1.jsonl" "$REC/intents.jsonl"
fi

echo "== 打分:卡 $CARD =="
$RUN "$EV" "$REC/intents.jsonl" --card "$CARD" --csv "$REC/${CARD}_score.csv"
