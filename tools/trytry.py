"""tools/trytry.py

这是一个“命令备忘录”，原本文件里是 bash 片段，直接 `python tools/trytry.py` 会报语法错。

现在改成：运行本文件会打印一份可直接复制执行的 bash 命令（默认按仓库路径自动推导）。

可选环境变量：
- `NUPLAN_DEVKIT_ROOT`：nuplan-devkit 路径（如果你的 navsim 依赖它）。
- `NAVSIM_EXP_ROOT`：navsim cache/experiment 根目录。
"""

from __future__ import annotations

import os


def _repo_root() -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
        repo_root = _repo_root()
        ddv2_root = os.path.join(repo_root, "DiffusionDriveV2")
        navsim_root = os.path.join(ddv2_root, "navsim")

        print("# ===================1) Caching=========================")
        print(f"REPO_ROOT=\"{repo_root}\"")
        print("DDV2_ROOT=\"$REPO_ROOT/DiffusionDriveV2\"")
        print("NAVSIM_ROOT=\"$DDV2_ROOT/navsim\"")
        print("export PYTHONPATH=\"$NAVSIM_ROOT:$DDV2_ROOT:$REPO_ROOT:${PYTHONPATH:-}\"")
        print("if [[ -n \"${NUPLAN_DEVKIT_ROOT:-}\" ]]; then export PYTHONPATH=\"$NUPLAN_DEVKIT_ROOT:$PYTHONPATH\"; fi")
        print("export NAVSIM_SKIP_MISSING=1")
        print("\n# cache dataset for training")
        print("python \"$NAVSIM_ROOT/planning/script/run_dataset_caching.py\" agent=diffusiondrivev2_rl_agent experiment_name=diffusiondrivev2_cache train_test_split=navtrain")

        print("\n# cache dataset for evaluation")
        print("python \"$NAVSIM_ROOT/planning/script/run_metric_caching.py\" train_test_split=navtest cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache")
        print("python \"$NAVSIM_ROOT/planning/script/run_metric_caching.py\" train_test_split=navtrain cache.cache_path=$NAVSIM_EXP_ROOT/train_pdm_cache")

        print("\n# cache dataset for fast evaluation (optional)")
        print("python \"$NAVSIM_ROOT/planning/script/run_dataset_caching.py\" agent=diffusiondrivev2_rl_agent experiment_name=diffusiondrivev2_cache train_test_split=navtest cache_path=$NAVSIM_EXP_ROOT/metric_feature_cache")

        print("\n# ===================2) Evaluating=========================")
        print("export TRAIN_TEST_SPLIT=navtest")
        print("export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print("export CUDA_VISIBLE_DEVICES=0,1,2,3")
        print("\n# fast evaluation")
        print("python \"$NAVSIM_ROOT/planning/script/run_pdm_score_fast.py\" agent=diffusiondrivev2_sel_agent experiment_name=diffusiondrivev2_agent_eval train_test_split=$TRAIN_TEST_SPLIT agent.checkpoint_path=\"$DDV2_ROOT/ckpt/diffusiondrivev2_sel.ckpt\" +metric_cache_path=\"$NAVSIM_EXP_ROOT/metric_cache/\" +test_cache_path=\"$NAVSIM_EXP_ROOT/metric_feature_cache/\"")

        print("\n# non-fast evaluation")
        print("python \"$NAVSIM_ROOT/planning/script/run_pdm_score.py\" agent=diffusiondrivev2_sel_agent experiment_name=diffusiondrivev2_agent_eval worker=sequential train_test_split=$TRAIN_TEST_SPLIT agent.checkpoint_path=\"$DDV2_ROOT/ckpt/diffusiondrivev2_sel.ckpt\" metric_cache_path=\"$NAVSIM_EXP_ROOT/metric_cache/\"")


if __name__ == "__main__":
        main()
