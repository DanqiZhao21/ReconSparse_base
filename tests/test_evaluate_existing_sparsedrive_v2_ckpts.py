from __future__ import annotations

from pathlib import Path

import yaml

from tools.smalltool.evaluateCache.evaluate_existing_sparsedrive_v2_ckpts import (
    discover_unique_scenarios,
    scenario_output_name,
)


def _write_scenario(path: Path, *, scene_name: str, mode: str) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "scene_name": scene_name,
                "mode": mode,
                "plan_list": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_scenario_output_name_matches_hugsim_scene_mode_output_dir(tmp_path: Path) -> None:
    scenario_path = tmp_path / "zscene-0013-medium-front-car-cutin2.yaml"
    _write_scenario(scenario_path, scene_name="scene-0013", mode="medium_01")

    assert scenario_output_name(scenario_path) == "scene-0013_medium_01"


def test_discover_unique_scenarios_deduplicates_same_scene_mode(tmp_path: Path) -> None:
    first = tmp_path / "scene-0013-medium-front-car-cutin.yaml"
    duplicate = tmp_path / "zscene-0013-medium-front-car-cutin.yaml"
    second = tmp_path / "zscene-0013-medium-front-car-cutin2.yaml"
    _write_scenario(first, scene_name="scene-0013", mode="medium_00")
    _write_scenario(duplicate, scene_name="scene-0013", mode="medium_00")
    _write_scenario(second, scene_name="scene-0013", mode="medium_01")

    discovered = discover_unique_scenarios(tmp_path)

    assert [path.name for path in discovered] == [
        "scene-0013-medium-front-car-cutin.yaml",
        "zscene-0013-medium-front-car-cutin2.yaml",
    ]
