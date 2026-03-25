#!/usr/bin/env bash
set -euo pipefail

# Repo-root relative runner for generate_video_sparsedrive.py
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

# -----------------------------------------------------------------------------
# User-tunable defaults (override by passing CLI args)
# -----------------------------------------------------------------------------
DEFAULT_SCENE_ID=${DEFAULT_SCENE_ID:-2}
DEFAULT_DURATION_S=${DEFAULT_DURATION_S:-}
DEFAULT_START_FRAME=${DEFAULT_START_FRAME:-0}
DEFAULT_STEP_FRAMES=${DEFAULT_STEP_FRAMES:-5}
DEFAULT_CUDA=${DEFAULT_CUDA:-0}
DEFAULT_MODE_SELECT=${DEFAULT_MODE_SELECT:-greedy}

DEFAULT_CONFIG=${DEFAULT_CONFIG:-"$REPO_ROOT/SparseDrive/projects/configs/sparsedrive_small_stage2.py"}
DEFAULT_CKPT=${DEFAULT_CKPT:-"$REPO_ROOT/SparseDrive/ckpt/sparsedrive_stage2.pth"}

# Optional fixed output path; empty -> auto naming
DEFAULT_OUT=${DEFAULT_OUT:-""}
DEFAULT_TRAJ_CSV=${DEFAULT_TRAJ_CSV:-""}

# Runtime env for CUDA/JIT extensions
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Keep torch JIT artifacts under repo
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/.cache/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"

# Ensure repo and SparseDrive are importable
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/SparseDrive:${PYTHONPATH:-}"

ORIG_ARGS=("$@")

EFFECTIVE_SCENE_ID="$DEFAULT_SCENE_ID"
EFFECTIVE_CKPT="$DEFAULT_CKPT"
EFFECTIVE_CONFIG="$DEFAULT_CONFIG"
USER_OUT=""
USER_TRAJ_CSV=""

# Parse --k=v forms
for a in "${ORIG_ARGS[@]}"; do
  case "$a" in
    --scene=*) EFFECTIVE_SCENE_ID="${a#*=}" ;;
    --ckpt=*) EFFECTIVE_CKPT="${a#*=}" ;;
    --config=*) EFFECTIVE_CONFIG="${a#*=}" ;;
    --out=*) USER_OUT="${a#*=}" ;;
    --traj-csv=*) USER_TRAJ_CSV="${a#*=}" ;;
  esac
done

# Parse --k v forms
for ((i=0; i<${#ORIG_ARGS[@]}; i++)); do
  if [[ "${ORIG_ARGS[$i]}" == "--scene" ]] && (( i+1 < ${#ORIG_ARGS[@]} )); then
    EFFECTIVE_SCENE_ID="${ORIG_ARGS[$((i+1))]}"
  fi
  if [[ "${ORIG_ARGS[$i]}" == "--ckpt" ]] && (( i+1 < ${#ORIG_ARGS[@]} )); then
    EFFECTIVE_CKPT="${ORIG_ARGS[$((i+1))]}"
  fi
  if [[ "${ORIG_ARGS[$i]}" == "--config" ]] && (( i+1 < ${#ORIG_ARGS[@]} )); then
    EFFECTIVE_CONFIG="${ORIG_ARGS[$((i+1))]}"
  fi
  if [[ "${ORIG_ARGS[$i]}" == "--out" ]] && (( i+1 < ${#ORIG_ARGS[@]} )); then
    USER_OUT="${ORIG_ARGS[$((i+1))]}"
  fi
  if [[ "${ORIG_ARGS[$i]}" == "--traj-csv" ]] && (( i+1 < ${#ORIG_ARGS[@]} )); then
    USER_TRAJ_CSV="${ORIG_ARGS[$((i+1))]}"
  fi
done

EFFECTIVE_SCENE_PADDED=$(printf "%03d" "${EFFECTIVE_SCENE_ID}")
TS=$(date +%Y%m%d-%H%M%S)
MODEL_NAME=$(basename "$EFFECTIVE_CKPT")
MODEL_NAME="${MODEL_NAME%.*}"

AUTO_OUT="$REPO_ROOT/outputs/visualize/scene${EFFECTIVE_SCENE_PADDED}_${MODEL_NAME}_${TS}_sparsedrive_rollout.mp4"
AUTO_TRAJ_CSV="$REPO_ROOT/outputs/visualize/scene${EFFECTIVE_SCENE_PADDED}_${MODEL_NAME}_${TS}_sparsedrive_plan_frontframe.csv"

FINAL_OUT="$USER_OUT"
if [[ -z "$FINAL_OUT" ]]; then
  if [[ -n "$DEFAULT_OUT" ]]; then
    FINAL_OUT="$DEFAULT_OUT"
  else
    FINAL_OUT="$AUTO_OUT"
  fi
fi

FINAL_TRAJ_CSV="$USER_TRAJ_CSV"
if [[ -z "$FINAL_TRAJ_CSV" ]]; then
  if [[ -n "$DEFAULT_TRAJ_CSV" ]]; then
    FINAL_TRAJ_CSV="$DEFAULT_TRAJ_CSV"
  else
    FINAL_TRAJ_CSV="$AUTO_TRAJ_CSV"
  fi
fi

CMD=(
  python "$REPO_ROOT/tools/smalltool/visualize/generate_video_sparsedrive.py"
  --scene "$EFFECTIVE_SCENE_ID"
  --config "$EFFECTIVE_CONFIG"
  --ckpt "$EFFECTIVE_CKPT"
  --out "$FINAL_OUT"
  --traj-csv "$FINAL_TRAJ_CSV"
  --start-frame "$DEFAULT_START_FRAME"
  --step-frames "$DEFAULT_STEP_FRAMES"
  --cuda "$DEFAULT_CUDA"
  --mode-select "$DEFAULT_MODE_SELECT"
)

if [[ -n "${DEFAULT_DURATION_S}" ]]; then
  CMD+=(--duration-s "$DEFAULT_DURATION_S")
fi

CMD+=("$@")

echo "[generate_video_sparsedrive.sh] REPO_ROOT=$REPO_ROOT"
echo "[generate_video_sparsedrive.sh] scene=$EFFECTIVE_SCENE_ID"
echo "[generate_video_sparsedrive.sh] config=$EFFECTIVE_CONFIG"
echo "[generate_video_sparsedrive.sh] ckpt=$EFFECTIVE_CKPT"
echo "[generate_video_sparsedrive.sh] out=$FINAL_OUT"
echo "[generate_video_sparsedrive.sh] traj_csv=$FINAL_TRAJ_CSV"

auto_mkdir_dir=$(dirname "$FINAL_OUT")
mkdir -p "$auto_mkdir_dir"
auto_mkdir_csv_dir=$(dirname "$FINAL_TRAJ_CSV")
mkdir -p "$auto_mkdir_csv_dir"

exec "${CMD[@]}"

# Usage examples:
# bash tools/smalltool/visualize/generate_video_sparsedrive.sh --scene=99
# bash tools/smalltool/visualize/generate_video_sparsedrive.sh --scene 99 --duration-s 20 --cuda 0
# bash tools/smalltool/visualize/generate_video_sparsedrive.sh --scene 99 --ckpt /abs/path/to/sparsedrive_stage2.pth
