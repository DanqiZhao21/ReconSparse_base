from pathlib import Path

from framework.utils.repo_paths import resolve_hugsim_path, resolve_hugsim_root


def test_resolve_hugsim_path_prefers_hugsim_root_for_relative_paths(monkeypatch, tmp_path):
    hugsim_root = tmp_path / "HUGSIM-ORI"
    scenario_dir = hugsim_root / "configs" / "scenarios" / "nuscenes"
    scenario_dir.mkdir(parents=True)
    monkeypatch.setenv("HUGSIM_ROOT", str(hugsim_root))

    assert resolve_hugsim_root() == str(hugsim_root)
    assert resolve_hugsim_path("configs/scenarios/nuscenes") == str(Path(scenario_dir))
