# #!/usr/bin/env bash
# set -euo pipefail

# # Repo root
# REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

# # -----------------------------------------------------------------------------
# # Defaults (can be overridden by CLI args)
# # -----------------------------------------------------------------------------
# DEFAULT_SCENE_ID=${DEFAULT_SCENE_ID:-99}
# DEFAULT_CSV=${DEFAULT_CSV:-"$REPO_ROOT/outputs/visualize/trajTransition/expert_ego_local_frame.csv"}
# DEFAULT_OUT_DIR=${DEFAULT_OUT_DIR:-"$REPO_ROOT/outputs/visualize/trajTransition"}
# DEFAULT_CUDA=${DEFAULT_CUDA:-0}
# DEFAULT_MAX_STEPS=${DEFAULT_MAX_STEPS:-""}
# DEFAULT_EXPERT_HIGH=${DEFAULT_EXPERT_HIGH:-0}

# # -----------------------------------------------------------------------------
# # Runtime environment (match existing working scripts)
# # -----------------------------------------------------------------------------
# export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
# export PATH="$CUDA_HOME/bin:${PATH:-}"
# export CPATH="$CUDA_HOME/include:${CPATH:-}"
# export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
# export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# # Keep torch JIT extension builds in repo-local cache.
# export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$REPO_ROOT/.cache/torch_extensions}"
# mkdir -p "$TORCH_EXTENSIONS_DIR"

# # Ensure project imports resolve.
# export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/DiffusionDriveV2:$REPO_ROOT/DiffusionDriveV2/navsim:${PYTHONPATH:-}"
# if [[ -n "${NUPLAN_DEVKIT_ROOT:-}" ]]; then
#   export PYTHONPATH="$NUPLAN_DEVKIT_ROOT:$PYTHONPATH"
# fi

# # Optional switch to clear stale nvdiffrast JIT artifacts.
# CLEAN_JIT_CACHE=0

# # -----------------------------------------------------------------------------
# # Parse user args (support --k=v and --k v)
# # -----------------------------------------------------------------------------
# EFFECTIVE_SCENE_ID="$DEFAULT_SCENE_ID"
# EFFECTIVE_CSV="$DEFAULT_CSV"
# EFFECTIVE_OUT_DIR="$DEFAULT_OUT_DIR"
# EFFECTIVE_CUDA="$DEFAULT_CUDA"
# EFFECTIVE_MAX_STEPS="$DEFAULT_MAX_STEPS"
# EFFECTIVE_EXPERT_HIGH="$DEFAULT_EXPERT_HIGH"

# ORIG_ARGS=("$@")
# PASSTHROUGH_ARGS=()

# for a in "${ORIG_ARGS[@]}"; do
#   case "$a" in
#     --scene=*) EFFECTIVE_SCENE_ID="${a#*=}" ;;
#     --csv=*) EFFECTIVE_CSV="${a#*=}" ;;
#     --out-dir=*) EFFECTIVE_OUT_DIR="${a#*=}" ;;
#     --cuda=*) EFFECTIVE_CUDA="${a#*=}" ;;
#     --max-steps=*) EFFECTIVE_MAX_STEPS="${a#*=}" ;;
#     --expert-high) EFFECTIVE_EXPERT_HIGH=1 ;;
#     --clean-jit-cache) CLEAN_JIT_CACHE=1 ;;
#     *) PASSTHROUGH_ARGS+=("$a") ;;
#   esac
# done

# for ((i=0; i<${#ORIG_ARGS[@]}; i++)); do
#   if [[ "${ORIG_ARGS[$i]}" == "--scene" ]] && (( i+1 < ${#ORIG_ARGS[@]} )); then
#     EFFECTIVE_SCENE_ID="${ORIG_ARGS[$((i+1))]}"
#   fi
#   if [[ "${ORIG_ARGS[$i]}" == "--csv" ]] && (( i+1 < ${#ORIG_ARGS[@]} )); then
#     EFFECTIVE_CSV="${ORIG_ARGS[$((i+1))]}"
#   fi
#   if [[ "${ORIG_ARGS[$i]}" == "--out-dir" ]] && (( i+1 < ${#ORIG_ARGS[@]} )); then
#     EFFECTIVE_OUT_DIR="${ORIG_ARGS[$((i+1))]}"
#   fi
#   if [[ "${ORIG_ARGS[$i]}" == "--cuda" ]] && (( i+1 < ${#ORIG_ARGS[@]} )); then
#     EFFECTIVE_CUDA="${ORIG_ARGS[$((i+1))]}"
#   fi
#   if [[ "${ORIG_ARGS[$i]}" == "--max-steps" ]] && (( i+1 < ${#ORIG_ARGS[@]} )); then
#     EFFECTIVE_MAX_STEPS="${ORIG_ARGS[$((i+1))]}"
#   fi
#   if [[ "${ORIG_ARGS[$i]}" == "--expert-high" ]]; then
#     EFFECTIVE_EXPERT_HIGH=1
#   fi
#   if [[ "${ORIG_ARGS[$i]}" == "--clean-jit-cache" ]]; then
#     CLEAN_JIT_CACHE=1
#   fi
# done

# if [[ "$CLEAN_JIT_CACHE" == "1" ]]; then
#   find "$TORCH_EXTENSIONS_DIR" -maxdepth 1 -type d -name "nvdiffrast*" -print -exec rm -rf {} +
# fi

# mkdir -p "$EFFECTIVE_OUT_DIR"

# CMD=(
#   python "$REPO_ROOT/tools/smalltool/visualize/replay_expert_vs_csv_video.py"
#   --scene "$EFFECTIVE_SCENE_ID"
#   --csv "$EFFECTIVE_CSV"
#   --out-dir "$EFFECTIVE_OUT_DIR"
#   --cuda "$EFFECTIVE_CUDA"
# )

# if [[ -n "$EFFECTIVE_MAX_STEPS" ]]; then
#   CMD+=(--max-steps "$EFFECTIVE_MAX_STEPS")
# fi

# if [[ "$EFFECTIVE_EXPERT_HIGH" == "1" ]]; then
#   CMD+=(--expert-high)
# fi

# CMD+=("${PASSTHROUGH_ARGS[@]}")

# echo "[run_replay_expert_vs_csv_video] REPO_ROOT=$REPO_ROOT"
# echo "[run_replay_expert_vs_csv_video] TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
# echo "[run_replay_expert_vs_csv_video] CUDA_HOME=$CUDA_HOME"
# echo "[run_replay_expert_vs_csv_video] scene=$EFFECTIVE_SCENE_ID csv=$EFFECTIVE_CSV"

# echo "[run_replay_expert_vs_csv_video] cmd: ${CMD[*]}"
# exec "${CMD[@]}"


#!/usr/bin/env bash
set -euo pipefail

# Repo root
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

# ----------------------------------------------------------------------
# Defaults (can be overridden by CLI args)
# ----------------------------------------------------------------------
DEFAULT_SCENE_ID=${DEFAULT_SCENE_ID:-99}
DEFAULT_CUDA=${DEFAULT_CUDA:-0}
DEFAULT_MAX_STEPS=${DEFAULT_MAX_STEPS:-""}
DEFAULT_EXPERT_HIGH=${DEFAULT_EXPERT_HIGH:-0}

# ----------------------------------------------------------------------
# Runtime environment
# ----------------------------------------------------------------------
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:${PATH:-}"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
# export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$REPO_ROOT/.cache/torch_extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/DiffusionDriveV2:$REPO_ROOT/DiffusionDriveV2/navsim:${PYTHONPATH:-}"
if [[ -n "${NUPLAN_DEVKIT_ROOT:-}" ]]; then
  export PYTHONPATH="$NUPLAN_DEVKIT_ROOT:$PYTHONPATH"
fi

CLEAN_JIT_CACHE=0

# ----------------------------------------------------------------------
# Parse user args
# ----------------------------------------------------------------------
EFFECTIVE_SCENE_ID="$DEFAULT_SCENE_ID"
EFFECTIVE_CUDA="$DEFAULT_CUDA"
EFFECTIVE_MAX_STEPS="$DEFAULT_MAX_STEPS"
EFFECTIVE_EXPERT_HIGH="$DEFAULT_EXPERT_HIGH"

ORIG_ARGS=("$@")
PASSTHROUGH_ARGS=()

for a in "${ORIG_ARGS[@]}"; do
  case "$a" in
    --scene=*) EFFECTIVE_SCENE_ID="${a#*=}" ;;
    --cuda=*) EFFECTIVE_CUDA="${a#*=}" ;;
    --max-steps=*) EFFECTIVE_MAX_STEPS="${a#*=}" ;;
    --expert-high) EFFECTIVE_EXPERT_HIGH=1 ;;
    --clean-jit-cache) CLEAN_JIT_CACHE=1 ;;
    *) PASSTHROUGH_ARGS+=("$a") ;;
  esac
done

# support --scene 77 style
for ((i=0; i<${#ORIG_ARGS[@]}; i++)); do
  if [[ "${ORIG_ARGS[$i]}" == "--scene" ]] && (( i+1 < ${#ORIG_ARGS[@]} )); then
    EFFECTIVE_SCENE_ID="${ORIG_ARGS[$((i+1))]}"
  fi
  if [[ "${ORIG_ARGS[$i]}" == "--cuda" ]] && (( i+1 < ${#ORIG_ARGS[@]} )); then
    EFFECTIVE_CUDA="${ORIG_ARGS[$((i+1))]}"
  fi
  if [[ "${ORIG_ARGS[$i]}" == "--max-steps" ]] && (( i+1 < ${#ORIG_ARGS[@]} )); then
    EFFECTIVE_MAX_STEPS="${ORIG_ARGS[$((i+1))]}"
  fi
  if [[ "${ORIG_ARGS[$i]}" == "--expert-high" ]]; then
    EFFECTIVE_EXPERT_HIGH=1
  fi
  if [[ "${ORIG_ARGS[$i]}" == "--clean-jit-cache" ]]; then
    CLEAN_JIT_CACHE=1
  fi
done

if [[ "$CLEAN_JIT_CACHE" == "1" ]]; then
  find "$TORCH_EXTENSIONS_DIR" -maxdepth 1 -type d -name "nvdiffrast*" -print -exec rm -rf {} +
fi

# ----------------------------------------------------------------------
# Scene-dependent paths
# ----------------------------------------------------------------------
SCENE_STR=$(printf "%03d" "$EFFECTIVE_SCENE_ID")
EFFECTIVE_OUT_DIR="$REPO_ROOT/outputs/visualize/trajTransition-scene$SCENE_STR"
EFFECTIVE_CSV="$EFFECTIVE_OUT_DIR/expert_ego_local_frame.csv"

mkdir -p "$EFFECTIVE_OUT_DIR"

# ----------------------------------------------------------------------
# Build python command
# ----------------------------------------------------------------------
CMD=(
  python "$REPO_ROOT/tools/smalltool/visualize/replay_expert_vs_csv_video.py"
  --scene "$EFFECTIVE_SCENE_ID"
  --csv "$EFFECTIVE_CSV"
  --out-dir "$EFFECTIVE_OUT_DIR"
  --cuda "$EFFECTIVE_CUDA"
)

if [[ -n "$EFFECTIVE_MAX_STEPS" ]]; then
  CMD+=(--max-steps "$EFFECTIVE_MAX_STEPS")
fi
if [[ "$EFFECTIVE_EXPERT_HIGH" == "1" ]]; then
  CMD+=(--expert-high)
fi

CMD+=("${PASSTHROUGH_ARGS[@]}")

echo "[run_replay_expert_vs_csv_video] REPO_ROOT=$REPO_ROOT"
echo "[run_replay_expert_vs_csv_video] TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
echo "[run_replay_expert_vs_csv_video] CUDA_HOME=$CUDA_HOME"
echo "[run_replay_expert_vs_csv_video] scene=$EFFECTIVE_SCENE_ID csv=$EFFECTIVE_CSV"

echo "[run_replay_expert_vs_csv_video] cmd: ${CMD[*]}"
exec "${CMD[@]}"