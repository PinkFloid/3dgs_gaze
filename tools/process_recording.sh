#!/usr/bin/env bash
# One-command processing of a Pupil Capture recording:
#   localize -> gaze precision stamps -> continuous gaze mapping (bias-corrected)
#   -> cone-posterior object assignment -> overlay video
#
# Usage:
#   tools/process_recording.sh ~/recordings/2026_07_05/002 [--skip-video]
#
# All outputs land inside the recording directory:
#   poses.jsonl                     T_world_cam per localized frame
#   gaze_precision.json             per-recording gaze bias/sigma/drift (tag stares)
#   world_fixations.json            world-space fixations (continuous clustering)
#   world_fixations_objects.json    per-fixation object posteriors (cone mode)
#   wfix/                           one annotated frame per fixation
#   gaze_objects_overlay.mp4        gaze cross + named-instance 3D boxes + verdicts
set -euo pipefail

REC="$(realpath "${1:?usage: process_recording.sh <recording_dir> [--skip-video]}")"
SKIP_VIDEO="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV="$HOME/miniconda3/envs/nerfstudio"
export PATH="$ENV/bin:$PATH" CUDA_HOME="$ENV" \
  CC="$ENV/bin/x86_64-conda-linux-gnu-gcc" CXX="$ENV/bin/x86_64-conda-linux-gnu-g++" \
  TORCH_CUDA_ARCH_LIST="5.2"
PY="$ENV/bin/python"
TAGS="$ROOT/world_size/tags_world.json"

echo "=== [1/5] localization (tags -> T_world_cam) ==="
"$PY" "$ROOT/tools/pupil_localizer.py" --recording "$REC" --tags "$TAGS" --log "$REC/poses.jsonl"

echo "=== [2/5] gaze precision stamps (tag-stare bias/sigma) ==="
"$PY" "$ROOT/tools/gaze_precision.py" --recording "$REC" --poses "$REC/poses.jsonl" \
  --tags "$TAGS" \
  || echo "WARN: no usable tag stare; continuing without bias correction (sigma default)"

echo "=== [3/5] continuous gaze -> world fixations (bias-corrected if stamped) ==="
"$PY" "$ROOT/tools/gaze_to_world.py" --recording "$REC" --poses "$REC/poses.jsonl" \
  --continuous --annotate-dir "$REC/wfix"

echo "=== [4/5] object assignment (gaze-cone posterior) ==="
"$PY" "$ROOT/tools/gaze_object.py" --cone --fixations "$REC/world_fixations.json"

if [[ "$SKIP_VIDEO" != "--skip-video" ]]; then
  echo "=== [5/5] overlay video ==="
  "$PY" "$ROOT/tools/gaze_video.py" --recording "$REC" \
    --objects "$REC/world_fixations_objects.json" \
    --poses "$REC/poses.jsonl" \
    --out "$REC/gaze_objects_overlay.mp4"
else
  echo "=== [5/5] skipped (--skip-video) ==="
fi

echo "=== done: $REC ==="
