#!/usr/bin/env bash
set -euo pipefail

# Batch-generate environment snapshots (env_cache.json) for scenes.
# Uses tools/smalltool/NuscenesEnvSnapForReward/build_metrics_cache.py to write assets/nus/data/<scene>/env_cache.json
# Skips scenes without assets/nus/data/<scene>/token_frame_map.json
#
# Configurable via env vars:
#   ROOT=/OpenDataset/nuscenes/nuscenes/v1.0-trainval
#   VERSION=v1.0-trainval
#   START=0
#   END=835
#   ROI=80.0          # ROI radius in meters
#   CL_RES=1.0        # Centerline discretization resolution (m)
#   VERBOSE=1         # Set to 1 for verbose per-frame warnings
#
# Example:
#   ROOT=/OpenDataset/nuscenes/nuscenes/v1.0-trainval VERSION=v1.0-trainval \
#   START=0 END=835 ROI=80 CL_RES=1.0 VERBOSE=1 \
#   bash tools/smalltool/NuscenesEnvSnapForReward/build_env_caches.sh

ROOT=${ROOT:-/OpenDataset/nuscenes/nuscenes/v1.0-trainval}
VERSION=${VERSION:-v1.0-trainval}
START=${START:-0}
END=${END:-835}
ROI=${ROI:-80.0}
CL_RES=${CL_RES:-1.0}
VERBOSE=${VERBOSE:-0}

# Resolve repo root for consistent execution
REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO_ROOT"

python_bin=${PYTHON:-python}

echo "Generating environment snapshots from scene ${START} to ${END}"
for ((i=START; i<=END; i++)); do
  scene_dir="assets/nus/data/$(printf "%03d" "$i")"
  token_map="${scene_dir}/token_frame_map.json"
  if [[ ! -f "$token_map" ]]; then
    echo "[skip] scene $(printf "%03d" "$i") has no token_frame_map.json"
    continue
  fi
  echo "[run] scene $(printf "%03d" "$i")"
  if [[ "$VERBOSE" == "1" ]]; then
    "$python_bin" tools/smalltool/NuscenesEnvSnapForReward/build_metrics_cache.py --scene "$i" --root "$ROOT" --version "$VERSION" --roi "$ROI" --cl_res "$CL_RES" --verbose || true
  else
    "$python_bin" tools/smalltool/NuscenesEnvSnapForReward/build_metrics_cache.py --scene "$i" --root "$ROOT" --version "$VERSION" --roi "$ROI" --cl_res "$CL_RES" || true
  fi
  # Brief pause to avoid hammering filesystem
  sleep 0.1
done

echo "Done. Check env_cache.json under assets/nus/data/<scene>/"
