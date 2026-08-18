#!/usr/bin/env bash
# score_card.sh <recording_dir> <card_id> -- E1 卡打分:裸跑(无偏置修正)+ eval
# 用户裁定(2026-08-18):不跑偏置三稿——8/16 实测裸跑与修正取优持平且更简单;
# 墙 tag 书签照录(留档可事后复查),但打分不吃它。
set -euo pipefail
REC=${1:?用法: score_card.sh <recording_dir> <card_id>}
CARD=${2:?用法: score_card.sh <recording_dir> <card_id>}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN="conda run --no-capture-output -n nerfstudio python -u"

$RUN "$ROOT/Eye_Tracker/tools/gaze_live.py" --replay "$REC" --headless \
    --on-tag-deg 0 --log "$REC/intents.jsonl" 2>&1 | grep -E "done:" || true
$RUN "$ROOT/Eye_Tracker/tools/eval_e1.py" "$REC/intents.jsonl" \
    --card "$CARD" --csv "$REC/${CARD}_score.csv"
