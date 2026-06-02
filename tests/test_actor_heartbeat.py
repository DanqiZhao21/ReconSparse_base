import time

import torch


def test_periodic_actor_heartbeat_writes_while_main_thread_is_blocked():
    from framework.runner.actor_heartbeat import PeriodicActorHeartbeat

    writes = []

    def writer(paths, actor_id, *, message=""):
        writes.append((paths, actor_id, message))
        return "heartbeat"

    heartbeat = PeriodicActorHeartbeat(
        paths="buffer",
        actor_id=7,
        interval_s=0.02,
        writer=writer,
    )

    heartbeat.start()
    try:
        heartbeat.beat("collect_single_step", 3, force=True)
        time.sleep(0.08)
    finally:
        heartbeat.stop()

    assert len(writes) >= 3
    assert writes[0] == ("buffer", 7, "collect_single_step step=3")
    assert writes[-1] == ("buffer", 7, "collect_single_step step=3")


def test_periodic_actor_heartbeat_disabled_when_interval_is_non_positive():
    from framework.runner.actor_heartbeat import PeriodicActorHeartbeat

    writes = []

    heartbeat = PeriodicActorHeartbeat(
        paths="buffer",
        actor_id=7,
        interval_s=0.0,
        writer=lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    heartbeat.start()
    heartbeat.beat("collect_single_step", force=True)
    time.sleep(0.03)
    heartbeat.stop()

    assert writes == []


def test_single_env_collection_reports_long_call_heartbeat_phases():
    from framework.rollout.collector import collect_single_env_shard

    beats = []

    class FakeAgent:
        def act(self, obs, *, eta, mode_idx, mode_select):
            return "action", torch.tensor(0.0), {}

        def supports_value_features(self):
            return False

    class FakeEnv:
        def step(self, action):
            return {"obs": 1}, 1.0, True, False, {}

        def reset(self):
            return {"obs": 2}, {}

    collect_single_env_shard(
        env=FakeEnv(),
        agent=FakeAgent(),
        obs={"obs": 0},
        horizon=1,
        eta=1.0,
        mode_idx=-1,
        mode_select="sample",
        actor_id=7,
        local_ver=0,
        shard_idx=0,
        store_obs=False,
        heartbeat_fn=lambda phase, step=None, **kwargs: beats.append((phase, step, kwargs.get("force", False))),
    )

    assert beats == [
        ("collect_single_step", 0, False),
        ("act_start", 0, True),
        ("act_done", 0, True),
        ("env_step_start", 0, True),
        ("env_step_done", 0, True),
        ("env_reset_start", 1, True),
        ("env_reset_done", 1, True),
    ]
