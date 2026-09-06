#!/usr/bin/env bash
# run_rejudge.sh -- 受控消融批(离线重判,同一份注视记录):
#   v2mass10   σ=1° 无面积归一(按原始锥质量份额排序)      <- 输入 intents_e4_v2s10.jsonl
#   v2noocc10  σ=1° 忽略遮挡的锥形查询(逐实例独立渲染)     <- 输入 intents_e4_v2s10.jsonl
#   v2noocc10ng 同上但去掉 0.5m 距离闸(敏感性)             <- 输入 intents_e4_v2s10.jsonl
#   v2noocc    σ=1.5° 忽略遮挡(补充对照)                  <- 输入 intents_e4_v2.jsonl
# 产物:<rec>/intents_e4_<cfg>.jsonl + <rec>/e4_<cfg>_score.csv;已有则跳过(可续跑)。
#   nohup ./Eye_Tracker/tools/run_rejudge.sh > <log> 2>&1 &
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RJ=$ROOT/Eye_Tracker/tools/rejudge.py
EV=$ROOT/Eye_Tracker/tools/eval_e1.py
RUN="conda run --no-capture-output -n nerfstudio python -u"
CK7=$ROOT/SceneRebuild/lab_result/splatfacto/2026-08-15_173942/nerfstudio_models/step-000029999.ckpt
CK8=$ROOT/SceneRebuild/lab_result/splatfacto/2026-08-18_125627/nerfstudio_models/step-000029999.ckpt
CK9=$ROOT/SceneRebuild/lab_result/splatfacto/2026-08-20_201525/nerfstudio_models/step-000029999.ckpt
A7=$ROOT/SceneRebuild/archive_envs/v7
A8=$ROOT/SceneRebuild/archive_envs/v8
A9=$ROOT/SceneRebuild/archive_envs/v9_rec
R=/home/liuchy/recordings
RECS="
$R/2026_08_16/000:e1:v7  $R/2026_08_16/001:e2:v7  $R/2026_08_16/s1:s1:v7
$R/2026_08_16/s2:s2:v7   $R/2026_08_16/s3:s3:v7
$R/2026_08_16/s4:s4:v8   $R/2026_08_16/s6:s6:v8   $R/2026_08_18/000:s7:v8
$R/2026_08_20/000:c1:v9  $R/2026_08_20/001:c2:v9  $R/2026_08_20/002:c4:v9
$R/2026_08_20/003:c4:v9  $R/2026_08_25/c1_1:c4:v9 $R/2026_08_25/c1_2:c4:v9
$R/2026_08_25/c1_3:c4:v9 $R/2026_08_25/u3:u3:v9   $R/2026_08_25/u1:u1:v9
"
# cfg:输入日志:flags
CFGS="v2vis10:intents_e4_v2s10.jsonl:--mode vis --rank capture
v2mass10:intents_e4_v2s10.jsonl:--mode full --rank mass
v2noocc10:intents_e4_v2s10.jsonl:--mode noocc --rank capture
v2noocc10ng:intents_e4_v2s10.jsonl:--mode noocc --rank capture --noocc-range-gate off
v2noocc:intents_e4_v2.jsonl:--mode noocc --rank capture"
for entry in $RECS; do
  IFS=: read -r rec card era <<< "$entry"
  case $era in
    v7) SEG=$A7; CK=$CK7;; v8) SEG=$A8; CK=$CK8;; v9) SEG=$A9; CK=$CK9;;
  esac
  while IFS= read -r cline; do
    cfg=${cline%%:*}; rest=${cline#*:}; inp=${rest%%:*}; flags=${rest#*:}
    log=$rec/intents_e4_${cfg}.jsonl; csv=$rec/e4_${cfg}_score.csv
    if [ ! -s "$rec/$inp" ]; then echo "  !! 缺输入 $rec/$inp"; continue; fi
    if [ ! -s "$log" ]; then
      echo "== rejudge $(basename $rec) [$cfg] $(date +%H:%M:%S)"
      $RUN "$RJ" "$rec/$inp" --seg-dir "$SEG" --ckpt "$CK" $flags --out "$log" 2>&1 | grep -E "wrote|check|!!|Error|error" \
          || echo "  !! rejudge 失败 $rec $cfg"
    fi
    if [ -s "$log" ] && [ ! -s "$csv" ]; then
      $RUN "$EV" "$log" --card "$card" --map-dir "$SEG" --csv "$csv" 2>/dev/null | grep 命中 \
          || echo "  !! eval 失败 $rec $cfg"
    fi
  done <<< "$CFGS"
done
echo "== rejudge 批跑完毕 $(date +%H:%M:%S)"
