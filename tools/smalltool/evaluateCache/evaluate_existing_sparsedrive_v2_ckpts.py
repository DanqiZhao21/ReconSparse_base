from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
HUGSIM_ROOT = Path(os.environ.get("HUGSIM_ROOT", REPO_ROOT / "third_party" / "HUGSIM-ORI")).resolve()
DEFAULT_HUGSIM_TEMPLATE = HUGSIM_ROOT / "configs" / "sim" / "nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml"
DEFAULT_SCENARIO_DIR = HUGSIM_ROOT / "configs" / "scenarios" / "nuscenes"
DEFAULT_EVAL_OUTPUT_ROOT = HUGSIM_ROOT / "outputs" / "evaluate-auto"
DEFAULT_EVAL_SLOTS = ["0:0", "1:1", "2:2", "3:3", "4:4", "5:5", "6:6", "7:7"]


def _load_hugsim_eval_runner():
    hugsim_path = str(HUGSIM_ROOT)
    if hugsim_path not in sys.path:
        sys.path.insert(0, hugsim_path)
    from utils import nuscenes_eval_runner as runner  # type: ignore

    return runner


def _slugify(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    if not slug:
        raise ValueError(f"Unable to derive slug from text={text!r}")
    return slug


def _scenario_output_name(scenario_path: Path) -> str:
    parts = scenario_path.stem.split("-")
    if len(parts) >= 4 and parts[0] == "scene":
        return f"scene-{parts[1]}_{'_'.join(parts[2:])}"
    return scenario_path.stem


def rewrite_hugsim_eval_config(
    *,
    template_config: Path,
    output_config: Path,
    ckpt_path: Path,
    output_prefix: str,
) -> None:
    payload = yaml.safe_load(template_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected yaml mapping in {template_config}")
    payload["sparsedrive_v2_ckpt"] = str(ckpt_path)
    payload["sparsedrive_v2_pretrain_ckpt"] = str(ckpt_path)
    payload["output_dir"] = str(output_prefix)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def build_eval_environment(
    *,
    base_env: dict[str, str] | None = None,
    hugsim_random_seed: str | int | None = None,
    default_eval_seed: bool = True,
) -> dict[str, str]:
    env = dict(base_env or {})
    if env.get("HUGSIM_DISABLE_DEFAULT_EVAL_SEED") == "1":
        return env
    seed_value = None if hugsim_random_seed is None else str(hugsim_random_seed)
    if seed_value is None and default_eval_seed:
        seed_value = "0"
    if seed_value is not None:
        env["HUGSIM_RANDOM_SEED"] = seed_value
        env["PYTHONHASHSEED"] = seed_value
    return env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing SparseDriveV2 checkpoints on HUGSIM-ORI NuScenes scenarios."
    )
    parser.add_argument("--ckpts", nargs="+", type=Path, required=True)
    parser.add_argument("--hugsim-template", type=Path, default=DEFAULT_HUGSIM_TEMPLATE)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--eval-output-root", type=Path, default=DEFAULT_EVAL_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--repeat-evals", type=int, default=2)
    parser.add_argument("--slots", nargs="+", default=list(DEFAULT_EVAL_SLOTS))
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runner = _load_hugsim_eval_runner()

    ckpt_paths = [path.resolve() for path in args.ckpts]
    missing_ckpts = [str(path) for path in ckpt_paths if not path.exists()]
    if missing_ckpts:
        raise FileNotFoundError(f"Missing checkpoints: {missing_ckpts}")

    scenario_dir = args.scenario_dir.resolve()
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

    scenario_paths = sorted(scenario_dir.glob("*.yaml"))
    if args.max_scenes is not None:
        scenario_paths = scenario_paths[: int(args.max_scenes)]
    if not scenario_paths:
        raise RuntimeError(f"No scenario YAMLs found under {scenario_dir}")

    run_name = args.run_name or f"sparsedrive_v2_hugsim_eval_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    run_root = args.eval_output_root.resolve() / run_name
    config_root = run_root / "input_configs"
    results_root = run_root / "results"
    summary_root = run_root / "summary"
    manifest_path = run_root / "manifest.json"

    tasks = []
    model_rows: list[dict[str, Any]] = []
    for ckpt_path in ckpt_paths:
        ckpt_stem = ckpt_path.stem
        ckpt_slug = _slugify(ckpt_stem)
        model_key = f"sparsedrive_v2_{ckpt_slug}"
        display_name = ckpt_stem
        output_dir_prefix = f"{ckpt_slug}_"
        official_output_dirname = f"{output_dir_prefix}sparsedrive_v2"
        input_config_path = config_root / f"{model_key}.yaml"
        rewrite_hugsim_eval_config(
            template_config=args.hugsim_template.resolve(),
            output_config=input_config_path,
            ckpt_path=ckpt_path,
            output_prefix=str(results_root / ckpt_slug / "placeholder_"),
        )
        model_rows.append(
            {
                "model_key": model_key,
                "display_name": display_name,
                "ckpt_path": str(ckpt_path),
                "input_config_path": str(input_config_path),
            }
        )
        for repeat_idx in range(1, int(args.repeat_evals) + 1):
            repeat_root = results_root / ckpt_slug / f"repeat_{repeat_idx:02d}"
            group_root = repeat_root / official_output_dirname
            for scenario_path in scenario_paths:
                tasks.append(
                    runner.TaskSpec(
                        model_key=model_key,
                        display_name=display_name,
                        repeat=repeat_idx,
                        scenario_path=scenario_path,
                        output_dir=group_root / _scenario_output_name(scenario_path),
                        ad="sparsedrive-v2",
                        config_path=input_config_path,
                        output_dir_prefix=output_dir_prefix,
                        official_output_dirname=official_output_dirname,
                        legacy_output_dirnames=(),
                    )
                )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "run_name": run_name,
                "scenario_dir": str(scenario_dir),
                "scenario_count": len(scenario_paths),
                "repeat_evals": int(args.repeat_evals),
                "retry_count": int(args.retry_count),
                "slots": list(args.slots),
                "summary_root": str(summary_root),
                "results_root": str(results_root),
                "models": model_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if args.dry_run:
        summary = runner.summarize_completed_results(tasks)
        runner.write_summary_artifacts(summary_root, summary)
        print(json.dumps({"run_root": str(run_root), "totals": summary["totals"]}, indent=2, sort_keys=True))
        return 0

    slots = runner._parse_slots(list(args.slots))
    batch_runner = runner.BatchRunner(
        repo_root=HUGSIM_ROOT,
        tasks=tasks,
        slots=slots,
        summary_root=summary_root,
        retry_count=int(args.retry_count),
    )
    os.environ.update(
        build_eval_environment(
            base_env=os.environ.copy(),
            default_eval_seed=True,
        )
    )
    return int(batch_runner.run())


if __name__ == "__main__":
    raise SystemExit(main())
