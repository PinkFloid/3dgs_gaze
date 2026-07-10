#!/usr/bin/env bash
# gaze_live.py wrapper: nerfstudio env python + gsplat JIT env vars baked in,
# so it works from any shell/conda env. All arguments pass through, e.g.:
#   tools/gaze_live.sh                      # live, window UI
#   tools/gaze_live.sh --publish 5581
#   tools/gaze_live.sh --replay ~/recordings/2026_07_09/000
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV="$HOME/miniconda3/envs/nerfstudio"
export PATH="$ENV/bin:$PATH" CUDA_HOME="$ENV" \
  CC="$ENV/bin/x86_64-conda-linux-gnu-gcc" CXX="$ENV/bin/x86_64-conda-linux-gnu-g++" \
  TORCH_CUDA_ARCH_LIST="5.2"
exec "$ENV/bin/python" "$DIR/gaze_live.py" "$@"
