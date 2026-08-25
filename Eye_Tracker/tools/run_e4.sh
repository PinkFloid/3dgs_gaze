#!/usr/bin/env bash
# run_e4.sh -- E4 回放消融批跑(Table II 数据):三个感知开关 × 全部 E1 录像。
#   nohup ./Eye_Tracker/tools/run_e4.sh > /tmp/e4_batch.log 2>&1 &
# 可断点续跑:已有 intents_e4_<cfg>.jsonl 的跳过回放,已有 csv 的跳过打分。
# 跨代:v7/v8 录像用 SceneRebuild/archive_envs/<era> + 对应 ckpt(先跑 collect 前先抽档)。
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
GL=$ROOT/Eye_Tracker/tools/gaze_live.py
EV=$ROOT/Eye_Tracker/tools/eval_e1.py
RUN="conda run --no-capture-output -n nerfstudio python -u"
CK7=$ROOT/SceneRebuild/lab_result/splatfacto/2026-08-15_173942/nerfstudio_models/step-000029999.ckpt
CK8=$ROOT/SceneRebuild/lab_result/splatfacto/2026-08-18_125627/nerfstudio_models/step-000029999.ckpt
CK9=$ROOT/SceneRebuild/lab_result/splatfacto/2026-08-20_201525/nerfstudio_models/step-000029999.ckpt
A7=$ROOT/SceneRebuild/archive_envs/v7
A8=$ROOT/SceneRebuild/archive_envs/v8
R=/home/liuchy/recordings

# rec:card:era(与 collect_e1.RECS 同源;u1 剔除条不跑)
RECS="
$R/2026_08_16/000:e1:v7  $R/2026_08_16/001:e2:v7  $R/2026_08_16/s1:s1:v7
$R/2026_08_16/s2:s2:v7   $R/2026_08_16/s3:s3:v7
$R/2026_08_16/s4:s4:v8   $R/2026_08_16/s6:s6:v8   $R/2026_08_18/000:s7:v8
$R/2026_08_20/000:c1:v9  $R/2026_08_20/001:c2:v9  $R/2026_08_20/002:c4:v9
$R/2026_08_20/003:c4:v9  $R/2026_08_25/c1_1:c4:v9 $R/2026_08_25/c1_2:c4:v9
$R/2026_08_25/c1_3:c4:v9 $R/2026_08_25/u3:u3:v9
"
# cfg:flags(full 基线即各录像现有 intents.jsonl,不重跑)
CFGS="votescope_all:--vote-scope all
priors_off:--priors off
cluster0:--cluster-deg 0"

for entry in $RECS; do
  IFS=: read -r rec card era <<< "$entry"
  case $era in
    v7) ENVF="--seg-dir $A7 --tags $A7/tags_world.json --ckpt $CK7";;
    v8) ENVF="--seg-dir $A8 --tags $A8/tags_world.json --ckpt $CK8";;
    v9) ENVF="--ckpt $CK9";;
  esac
  while IFS= read -r cline; do
    cfg=${cline%%:*}; flags=${cline#*:}
    log=$rec/intents_e4_${cfg}.jsonl
    csv=$rec/e4_${cfg}_score.csv
    if [ ! -s "$log" ]; then
      echo "== replay $(basename $rec) [$cfg] $(date +%H:%M:%S)"
      $RUN "$GL" --replay "$rec" --headless --on-tag-deg 0 $ENVF $flags \
          --log "$log" 2>&1 | grep -E "done:" || echo "  !! replay 失败 $rec $cfg"
    fi
    if [ -s "$log" ] && [ ! -s "$csv" ]; then
      mapdir=$([ "$era" = v9 ] && echo "$ROOT/SceneRebuild/lab_result/segmentation_sam" || echo "$([ "$era" = v7 ] && echo $A7 || echo $A8)")
      $RUN "$EV" "$log" --card "$card" --map-dir "$mapdir" --csv "$csv" 2>/dev/null | grep 命中 \
          || echo "  !! eval 失败 $rec $cfg"
    fi
  done <<< "$CFGS"
done
echo "== E4 批跑完毕 $(date +%H:%M:%S)"
