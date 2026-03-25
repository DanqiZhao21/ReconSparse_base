#!/usr/bin/env bash
set -euo pipefail

# Repo-root relative runner for generate_video_sparsedrive_v2.py
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EGOADS_ROOT="$REPO_ROOT/egoADs"
SPARSEDRIVE_V2_ROOT="$REPO_ROOT/SparseDriveV2"
if [[ -d "$EGOADS_ROOT/SparseDriveV2" ]]; then
  SPARSEDRIVE_V2_ROOT="$EGOADS_ROOT/SparseDriveV2"
fi

# -----------------------------------------------------------------------------
# User-tunable defaults (override by passing CLI args)
# -----------------------------------------------------------------------------
DEFAULT_SCENE_ID=${DEFAULT_SCENE_ID:-2}
DEFAULT_DURATION_S=${DEFAULT_DURATION_S:-}
DEFAULT_START_FRAME=${DEFAULT_START_FRAME:-0}
DEFAULT_STEP_FRAMES=${DEFAULT_STEP_FRAMES:-5}
DEFAULT_CUDA=${DEFAULT_CUDA:-0}
DEFAULT_MODE_SELECT=${DEFAULT_MODE_SELECT:-greedy}

DEFAULT_CKPT=${DEFAULT_CKPT:-"$SPARSEDRIVE_V2_ROOT/ckpt/sparsedrive_navsimv2.ckpt"}

# Optional fixed output path; empty -> auto naming
DEFAULT_OUT=${DEFAULT_OUT:-""}
DEFAULT_TRAJ_CSV=${DEFAULT_TRAJ_CSV:-""}
DEFAULT_TRAJ_PLOT=${DEFAULT_TRAJ_PLOT:-""}

# Runtime env for CUDA/JIT extensions
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Keep torch JIT artifacts under repo
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/.cache/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"

# Ensure repo and SparseDriveV2 are importable
export PYTHONPATH="$REPO_ROOT:$SPARSEDRIVE_V2_ROOT:${PYTHONPATH:-}"

ORIG_ARGS=("$@")

EFFECTIVE_SCENE_ID="$DEFAULT_SCENE_ID"
EFFECTIVE_CKPT="$DEFAULT_CKPT"
USER_OUT=""
USER_TRAJ_CSV=""
USER_TRAJ_PLOT=""

# Parse --k=v forms
for a in "${ORIG_ARGS[@]}"; do
  case "$a" in
    --scene=*) EFFECTIVE_SCENE_ID="${a#*=}" ;;
    --ckpt=*) EFFECTIVE_CKPT="${a#*=}" ;;
    --out=*) USER_OUT="${a#*=}" ;;
    --traj-csv=*) USER_TRAJ_CSV="${a#*=}" ;;
    --traj-plot=*) USER_TRAJ_PLOT="${a#*=}" ;;
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
  if [[ "${ORIG_ARGS[$i]}" == "--out" ]] && (( i+1 < ${#ORIG_ARGS[@]} )); then
    USER_OUT="${ORIG_ARGS[$((i+1))]}"
  fi
  if [[ "${ORIG_ARGS[$i]}" == "--traj-csv" ]] && (( i+1 < ${#ORIG_ARGS[@]} )); then
    USER_TRAJ_CSV="${ORIG_ARGS[$((i+1))]}"
  fi
  if [[ "${ORIG_ARGS[$i]}" == "--traj-plot" ]] && (( i+1 < ${#ORIG_ARGS[@]} )); then
    USER_TRAJ_PLOT="${ORIG_ARGS[$((i+1))]}"
  fi
done

EFFECTIVE_SCENE_PADDED=$(printf "%03d" "${EFFECTIVE_SCENE_ID}")
TS=$(date +%Y%m%d-%H%M%S)
MODEL_NAME=$(basename "$EFFECTIVE_CKPT")
MODEL_NAME="${MODEL_NAME%.*}"
AUTO_OUT_DIR="$REPO_ROOT/outputs/visualize/sparsedriveV2"

AUTO_OUT="$AUTO_OUT_DIR/scene${EFFECTIVE_SCENE_PADDED}_${MODEL_NAME}_${TS}_sparsedrivev2_rollout.mp4"
AUTO_TRAJ_CSV="$AUTO_OUT_DIR/scene${EFFECTIVE_SCENE_PADDED}_${MODEL_NAME}_${TS}_sparsedrivev2_plan_frontframe.csv"
AUTO_TRAJ_PLOT="$AUTO_OUT_DIR/scene${EFFECTIVE_SCENE_PADDED}_${MODEL_NAME}_${TS}_sparsedrivev2_expert_vs_ego_traj.svg"

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

FINAL_TRAJ_PLOT="$USER_TRAJ_PLOT"
if [[ -z "$FINAL_TRAJ_PLOT" ]]; then
  if [[ -n "$DEFAULT_TRAJ_PLOT" ]]; then
    FINAL_TRAJ_PLOT="$DEFAULT_TRAJ_PLOT"
  else
    FINAL_TRAJ_PLOT="$AUTO_TRAJ_PLOT"
  fi
fi

CMD=(
  python "$REPO_ROOT/tools/smalltool/visualize/generate_video_sparsedrive_v2.py"
  --scene "$EFFECTIVE_SCENE_ID"
  --ckpt "$EFFECTIVE_CKPT"
  --out "$FINAL_OUT"
  --traj-csv "$FINAL_TRAJ_CSV"
  --traj-plot "$FINAL_TRAJ_PLOT"
  --start-frame "$DEFAULT_START_FRAME"
  --step-frames "$DEFAULT_STEP_FRAMES"
  --cuda "$DEFAULT_CUDA"
  --mode-select "$DEFAULT_MODE_SELECT"
)

if [[ -n "${DEFAULT_DURATION_S}" ]]; then
  CMD+=(--duration-s "$DEFAULT_DURATION_S")
fi

CMD+=("$@")

echo "[generate_video_sparsedrive_v2.sh] REPO_ROOT=$REPO_ROOT"
echo "[generate_video_sparsedrive_v2.sh] scene=$EFFECTIVE_SCENE_ID"
echo "[generate_video_sparsedrive_v2.sh] ckpt=$EFFECTIVE_CKPT"
echo "[generate_video_sparsedrive_v2.sh] out=$FINAL_OUT"
echo "[generate_video_sparsedrive_v2.sh] traj_csv=$FINAL_TRAJ_CSV"
echo "[generate_video_sparsedrive_v2.sh] traj_plot=$FINAL_TRAJ_PLOT"

mkdir -p "$(dirname "$FINAL_OUT")"
mkdir -p "$(dirname "$FINAL_TRAJ_CSV")"
mkdir -p "$(dirname "$FINAL_TRAJ_PLOT")"

# ReconSimulator uses relative asset paths (assets/nus/...).
# Run from repo root to make those paths deterministic.
cd "$REPO_ROOT"
echo "[generate_video_sparsedrive_v2.sh] cwd=$(pwd)"

exec "${CMD[@]}"

# Usage examples:
# bash tools/smalltool/visualize/generate_video_sparsedrive_v2.sh --scene=99
# bash tools/smalltool/visualize/generate_video_sparsedrive_v2.sh --scene 99 --cuda 0
# bash tools/smalltool/visualize/generate_video_sparsedrive_v2.sh --scene 99 --duration-s 20
# bash tools/smalltool/visualize/generate_video_sparsedrive_v2.sh --scene 99 --ckpt /abs/path/to/sparsedrive_navsimv2.ckpt
