#!/bin/bash
# Training launcher with automatic resource cleanup
# Ensures clean starts and prevents training from getting stuck

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
REPO_ROOT="/root/clone/ReconDreamer-RL"
PYTHONPATH="${REPO_ROOT}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup_failed_actors() {
    local buffer_dir="$1"
    log_info "Checking for failed actors in ${buffer_dir}/actors/"

    if [ -d "${buffer_dir}/actors" ]; then
        shopt -s nullglob
        for actor_file in "${buffer_dir}/actors"/actor*.heartbeat; do
            shopt -u nullglob
            if [ -f "$actor_file" ]; then
                actor_id=$(basename "$actor_file" .heartbeat | sed 's/actor//')
                # Check if actor is stuck in "init" phase
                if grep -q "^message=init$" "$actor_file" 2>/dev/null; then
                    mtime=$(stat -c "%Y" "$actor_file")
                    current_time=$(date +%s)
                    age=$((current_time - mtime))

                    # If stuck in init for more than 120 seconds, kill it
                    if [ $age -gt 120 ]; then
                        log_warn "Actor ${actor_id} stuck in init phase for ${age}s"
                        # Kill any HUGSIM processes for this actor
                        pkill -f "scene-.*actor${actor_id}" 2>/dev/null || true
                    fi
                fi
            fi
        done
        shopt -u nullglob
    fi
}

cleanup_leaked_processes() {
    log_info "Cleaning up leaked processes..."

    # Kill HUGSIM FIFO runners
    pkill -f "hugsim_fifo_runner" 2>/dev/null || true
    killed=$?

    if [ $killed -eq 0 ]; then
        log_info "No leaked HUGSIM processes found"
    elif [ $killed -le 2 ]; then
        log_info "Killed ${killed} leaked HUGSIM process(es)"
    else
        log_warn "Killed ${killed} leaked HUGSIM processes"
    fi

    # Kill any zombie training processes
    pkill -9 -f "train_actor_learner.*actor.*actor" 2>/dev/null || true
}

check_training_conflict() {
    local config_file="$1"

    # Extract buffer directory from config
    buffer_dir=$(grep -A 50 "buffer_dir:" "$config_file" | head -1 | sed 's/.*buffer_dir: *\([^#]*\).*/\1/')
    buffer_dir="${buffer_dir:-outputs/actor_learner}"

    if [ -z "$buffer_dir" ]; then
        return
    fi

    # Check for existing training lock
    training_lock="${buffer_dir}/TRAINING_LOCK"
    if [ -f "$training_lock" ] || [ -d "${buffer_dir}/buffer" ]; then
        # Check for active training processes
        if pgrep -f "train_actor_learner.*${buffer_dir}" > /dev/null; then
            log_error "Training already running in ${buffer_dir}"
            log_error "Please stop existing training first or use a different buffer_dir"
            log_error "To force cleanup, run: pkill -9 -f 'train_actor_learner.*${buffer_dir}'"
            exit 1
        fi
    fi
}

launch_training() {
    local config_file="$1"

    log_info "Starting training with config: ${config_file}"

    # Check for training conflicts
    check_training_conflict "$config_file"

    # Clean up any leaked processes
    cleanup_leaked_processes

    # Launch training
    log_info "Launching training..."
    PYTHONPATH="${PYTHONPATH}" python -u "${REPO_ROOT}/script/train_actor_learner_v2.py" \
        --role orchestrator \
        --config "$config_file"
}

# Main entry point
if [ $# -lt 1 ]; then
    log_error "Usage: $0 <config_file.yaml>"
    log_error "Example: $0 script/configs/sparsedrive_v2/your_config.yaml"
    exit 1
fi

config_file="$1"
if [ ! -f "$config_file" ]; then
    log_error "Config file not found: ${config_file}"
    exit 1
fi

# Handle cleanup mode
if [ "$1" = "--cleanup" ]; then
    buffer_dir=$(grep -A 50 "buffer_dir:" "$config_file" | head -1 | sed 's/.*buffer_dir: *\([^#]*\).*/\1/')
    buffer_dir="${buffer_dir:-outputs/actor_learner}"

    log_info "Running cleanup for buffer directory: ${buffer_dir}"
    cleanup_leaked_processes
    cleanup_failed_actors "$buffer_dir"
    exit 0
fi

launch_training "$config_file"