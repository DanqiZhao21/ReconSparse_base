#====================================================
# cache dataset for evaluation ✔👌🎈
#========================================================
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DDV2_ROOT="$REPO_ROOT/DiffusionDriveV2"
NAVSIM_ROOT="$DDV2_ROOT/navsim"
export PYTHONPATH="$NAVSIM_ROOT:$DDV2_ROOT:$REPO_ROOT:${PYTHONPATH:-}"
if [[ -n "${NUPLAN_DEVKIT_ROOT:-}" ]]; then
	export PYTHONPATH="$NUPLAN_DEVKIT_ROOT:$PYTHONPATH"
fi

python "$NAVSIM_ROOT/planning/script/run_metric_caching.py" \
train_test_split=navtest cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
# train_test_split=navtest cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache

# cache dataset for calculating PDMS during training.
python "$NAVSIM_ROOT/planning/script/run_metric_caching.py" train_test_split=navtrain cache.cache_path=$NAVSIM_EXP_ROOT/train_pdm_cache