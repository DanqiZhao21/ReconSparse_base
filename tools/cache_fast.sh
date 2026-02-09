#====================================================
# cache dataset for fast evaluation  (optional) ✔👌🎈
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

python "$NAVSIM_ROOT/planning/script/run_dataset_caching.py" agent=diffusiondrivev2_rl_agent experiment_name=diffusiondrivev2_cache train_test_split=navtest cache_path=$NAVSIM_EXP_ROOT/metric_feature_cache

