import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "egoADs" / "SparseDriveV2"))

from navsim.planning.script.distributed_utils import merge_prediction_shards


def test_merge_prediction_shards_single_process_passthrough():
    predictions = [{"a": 1}, {"b": 2}]

    merged = merge_prediction_shards([predictions])

    assert merged == {"a": 1, "b": 2}


def test_merge_prediction_shards_multi_process_merges_all_dicts():
    shards = [
        [{"a": 1}, {"b": 2}],
        [{"c": 3}],
        [{"d": 4, "e": 5}],
    ]

    merged = merge_prediction_shards(shards)

    assert merged == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
