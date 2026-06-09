from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path("/root/clone/ReconDreamer-RL")
PIPELINE = REPO_ROOT / "script" / "run_train_eval_pipeline_hugsim_ori.sh"
LOG_ROOT = REPO_ROOT / "outputs" / "train_eval_queues" / "20260607_grpo_collision_aux"

CONFIGS = [
    REPO_ROOT / "script/configs/sparsedrive_v2/202606070001_HUGSM_grpo_only_closed_loop_steppath_hd_onlyGRPO_expectedprob_substeps1.yaml",
    REPO_ROOT / "script/configs/sparsedrive_v2/202606070002_HUGSM_reinforcepp_closed_loop_steppath_hd_collision_only_extreme_GRPOCraft_substeps1.yaml",
    REPO_ROOT / "script/configs/sparsedrive_v2/202606070003_HUGSM_reinforcepp_closed_loop_steppath_hd_collision_only_extreme_GRPOCraft_auxRiskDecel_substeps1.yaml",
]

TAGS = [
    "only_grpo",
    "collision_only_grpo",
    "collision_only_grpo_aux_risk_decel",
]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _processes_in_session(session_id: int) -> list[int]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,sid"], text=True)
    except Exception:
        return []
    pids: list[int] = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            sid = int(parts[1])
        except ValueError:
            continue
        if sid == int(session_id) and pid != os.getpid():
            pids.append(pid)
    return pids


def _cleanup_session(session_id: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = _processes_in_session(session_id)
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if pids:
            time.sleep(10 if sig == signal.SIGTERM else 2)


def _run_one_until_success(config_path: Path, tag: str) -> None:
    attempt = 1
    while True:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        log_path = LOG_ROOT / f"{tag}_attempt{attempt}_{_stamp()}.log"
        cmd = [
            str(PIPELINE),
            "--reinforcepp-config",
            str(config_path),
            "--reinforcepp-algo-tag",
            str(tag),
        ]
        print(f"[{_now()}] start tag={tag} attempt={attempt} cfg={config_path} log={log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"[cwd] {REPO_ROOT}\n")
            handle.write(f"[cmd] {' '.join(cmd)}\n")
            handle.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            try:
                code = proc.wait()
            except BaseException:
                _cleanup_session(proc.pid)
                raise
        if code == 0:
            _cleanup_session(proc.pid)
            print(f"[{_now()}] success tag={tag} attempt={attempt} log={log_path}", flush=True)
            return
        print(f"[{_now()}] failed tag={tag} attempt={attempt} code={code} log={log_path}", flush=True)
        _cleanup_session(proc.pid)
        attempt += 1
        time.sleep(60)


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[{_now()}] queue_start log_root={LOG_ROOT}", flush=True)
    for config_path, tag in zip(CONFIGS, TAGS, strict=True):
        _run_one_until_success(config_path, tag)
    print(f"[{_now()}] queue_complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
