import ast
from pathlib import Path


def _assignment_targets_size_lw(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad_lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Subscript):
            continue
        value = node.value
        if not (
            isinstance(value.value, ast.Name)
            and value.value.id == "ann"
            and isinstance(value.slice, ast.Constant)
            and value.slice.value == "size"
        ):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Tuple) or len(target.elts) < 2:
                continue
            first, second = target.elts[0], target.elts[1]
            if (
                isinstance(first, ast.Name)
                and isinstance(second, ast.Name)
                and first.id == "l"
                and second.id == "w"
            ):
                bad_lines.append(int(node.lineno))
    return bad_lines


def test_nuscenes_annotation_size_is_read_as_width_length_height() -> None:
    """nuScenes sample_annotation.size is [width, length, height], not [length, width, height]."""
    paths = [
        Path("framework/utils/build_metrics_cache.py"),
        Path("reconsimulator/envs/metrics.py"),
        Path("tools/smalltool/NuscenesEnvSnapForReward/build_metrics_cache.py"),
        Path("tools/smalltool/NuscenesEnvSnapForReward/get_info.py"),
        Path("tools/smalltool/NuscenesEnvSnapForReward/get_info_1.py"),
        Path("tools/smalltool/NuscenesEnvSnapForReward/get_info_2.py"),
    ]
    offenders = {
        str(path): _assignment_targets_size_lw(path)
        for path in paths
        if _assignment_targets_size_lw(path)
    }
    assert offenders == {}
