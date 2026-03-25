#!/usr/bin/env bash
set -euo pipefail

# Repo-root relative runner for generate_video.py
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

# -----------------------------------------------------------------------------
# User-tunable defaults (override by passing CLI args)
# -----------------------------------------------------------------------------

# 默认场景 id
DEFAULT_SCENE_ID=${DEFAULT_SCENE_ID:-2}
# 默认时长（秒）；设为空字符串表示跑到 done
DEFAULT_DURATION_S=${DEFAULT_DURATION_S:-18}
# 默认 ckpt 路径
# /root/clone/ReconDreamer-RL/DiffusionDriveV2/ckpt/diffusiondrivev2_sel.ckpt
# /root/clone/ReconDreamer-RL/outputs/actor_learner/weights/latest.ckpt
# $REPO_ROOT=/root/clone/ReconDreamer-RL
#一下为：自己训的；ddv2-sel;ddv2-rl;
# DEFAULT_CKPT=${DEFAULT_CKPT:-"$REPO_ROOT/outputs/weight/20260129_ppo_ver27_latest.ckpt"}
# DEFAULT_CKPT=${DEFAULT_CKPT:-"$REPO_ROOT/DiffusionDriveV2/ckpt/diffusiondrivev2_sel.ckpt"}
# DEFAULT_CKPT=${DEFAULT_CKPT:-"$REPO_ROOT/DiffusionDriveV2/ckpt/diffusiondrivev2_rl.ckpt"}

#'''
# self.plan_anchor_scorer_encoder = nn.Sequential(
#             *linear_relu_ln(d_model, 1, 1,2*512),
#             # nn.Linear(d_model, d_model),#NOTE:使用sel 模型的时候需要将这里的d_model改成512
#             nn.Linear(d_model, 512),
# '''



#TODO: 目前会报错missing keys
DEFAULT_CKPT=${DEFAULT_CKPT:-"$REPO_ROOT/DiffusionDriveV2/ckpt/diffusiondrivev2_rl.ckpt"}


# 默认模型名字（可不填；不填则从 ckpt 文件名自动推断）
DEFAULT_MODEL_NAME=${DEFAULT_MODEL_NAME:-""}
# 默认输出路径：
# - 设为空字符串表示自动生成 outputs/visualize/sceneXXX_<时间戳>.mp4
# - 也可以直接设成一个固定文件路径（必须以 .mp4 结尾）
DEFAULT_OUT=${DEFAULT_OUT:-""}
# CUDA headers/libs are needed for some JIT-compiled render extensions (e.g., nvdiffrast).
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Keep torch JIT build artifacts inside the repo (avoid clutter/permission issues).
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/.cache/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/DiffusionDriveV2:$REPO_ROOT/DiffusionDriveV2/navsim:${PYTHONPATH:-}"
if [[ -n "${NUPLAN_DEVKIT_ROOT:-}" ]]; then
  export PYTHONPATH="$NUPLAN_DEVKIT_ROOT:$PYTHONPATH"
fi

EFFECTIVE_SCENE_ID="$DEFAULT_SCENE_ID"
EFFECTIVE_CKPT="$DEFAULT_CKPT"
USER_OUT=""
USE_EXPERT=0
ORIG_ARGS=("$@")
for a in "${ORIG_ARGS[@]}"; do
  case "$a" in
    --scene=*) EFFECTIVE_SCENE_ID="${a#*=}" ;;
    --ckpt=*) EFFECTIVE_CKPT="${a#*=}" ;;
    --out=*) USER_OUT="${a#*=}" ;;
    --expert) USE_EXPERT=1 ;;
  esac
done

# 处理 --scene <id> / --out <path> 这种“分开写”的形式
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
  if [[ "${ORIG_ARGS[$i]}" == "--expert" ]]; then
    USE_EXPERT=1
  fi
done

EFFECTIVE_SCENE_PADDED=$(printf "%03d" "${EFFECTIVE_SCENE_ID}")
TS=$(date +%Y%m%d-%H%M%S)

# 模型名：优先用 DEFAULT_MODEL_NAME，其次从 ckpt 文件名推断
MODEL_NAME="$DEFAULT_MODEL_NAME"
if [[ "$USE_EXPERT" == "1" ]]; then
  MODEL_NAME="expert"
elif [[ -z "$MODEL_NAME" ]]; then
  CKPT_BASE=$(basename "$EFFECTIVE_CKPT")
  MODEL_NAME="${CKPT_BASE%.*}"
fi

AUTO_OUT="$REPO_ROOT/outputs/visualize/scene${EFFECTIVE_SCENE_PADDED}_${MODEL_NAME}_${TS}.mp4"

FINAL_OUT="$USER_OUT"
if [[ -z "$FINAL_OUT" ]]; then
  if [[ -n "$DEFAULT_OUT" ]]; then
    FINAL_OUT="$DEFAULT_OUT"
  else
    FINAL_OUT="$AUTO_OUT"
  fi
fi

CMD=(
  python "$REPO_ROOT/tools/smalltool/visualize/generate_video.py"
  --scene "$DEFAULT_SCENE_ID"
  --ckpt "$DEFAULT_CKPT"
  --out "$FINAL_OUT"
)

if [[ -n "${DEFAULT_DURATION_S}" ]]; then
  CMD+=(--duration-s "$DEFAULT_DURATION_S")
fi

CMD+=("$@")
exec "${CMD[@]}"



# '''
# 用法示例（Python 侧无需改动）：
# bash /root/clone/ReconDreamer-RL/tools/smalltool/visualize/generate_video.sh --scene=7
# '''
