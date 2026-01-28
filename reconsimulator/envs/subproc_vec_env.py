from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .rl_wrapper import RLReconEnv


@dataclass(frozen=True)
class SceneSamplingSpec:
    scene_ids: List[int]
    scene_sampling: str  # random | sequential
    ddp_seed: int
    rank: int
    worker_id: int

    start_mode: str  # zero | random | sequential
    allow_short_tail: bool
    start_min: int
    start_max: Optional[int]
    start_stride: Optional[int]
    max_steps: int


class SceneSamplingEnv:
    """RLReconEnv wrapper that samples (scene_id, start_frame) on each reset.

    This keeps subproc workers self-contained: the main process only needs to call
    reset()/step(), and each worker handles scene rotation & start_frame sampling.
    """

    def __init__(
        self,
        *,
        cuda: int,
        reward_cfg: Dict[str, Any] | None,
        debug: bool,
        spec: SceneSamplingSpec,
        render_w: int | None = None,
        render_h: int | None = None,
        step_frames: int | None = None,
    ) -> None:
        self._cuda = int(cuda)
        self._reward_cfg = reward_cfg or {}
        self._debug = bool(debug)
        self._spec = spec
        self._render_w = (int(render_w) if render_w is not None else None)
        self._render_h = (int(render_h) if render_h is not None else None)
        self._step_frames = (int(step_frames) if step_frames is not None else None)

        self._scene_ids: List[int] = list(spec.scene_ids) if len(spec.scene_ids) else [0]
        self._scene_sampling = str(spec.scene_sampling or "random").lower()

        # Per-worker deterministic state
        self._seq_idx = int(spec.worker_id)
        self._reset_counter = 0
        self._scene_start_cursor: Dict[int, int] = {}

        # Create once; scene switches via env.reset(scene=...)
        init_scene = int(self._scene_ids[0])
        self._env = RLReconEnv(
            cuda=self._cuda,
            scene=init_scene,
            reward_cfg=self._reward_cfg,
            debug=self._debug,
            render_w=self._render_w,
            render_h=self._render_h,
        )
        self._current_scene: int = int(init_scene)

        # Cache last (obs, info) so the main process can temporarily idle this worker
        # without advancing the environment (useful when collecting fixed #episodes).
        self._last_obs: Dict[str, np.ndarray] | None = None
        self._last_info: Dict[str, Any] | None = None

    @property
    def env(self) -> RLReconEnv:
        return self._env

    def _pick_scene_id(self) -> int:
        if len(self._scene_ids) == 0:
            return int(self._current_scene)
        if self._scene_sampling.startswith("seq"):
            sid = int(self._scene_ids[self._seq_idx % len(self._scene_ids)])
            self._seq_idx += 1
            return sid

        # Deterministic given (ddp_seed, rank, worker_id, reset_counter)
        seed = int(self._spec.ddp_seed) + int(self._spec.rank) * 100003 + int(self._spec.worker_id) * 1009 + int(self._reset_counter) * 97
        rng = np.random.RandomState(seed)
        return int(rng.choice(self._scene_ids))

    def _sample_start_frame(self, *, scene_id: int) -> int:
        mode = str(self._spec.start_mode or "zero").lower()
        if mode.startswith("zero"):
            return 0
        if (not mode.startswith("rand")) and (not mode.startswith("seq")):
            return 0

        try:
            final_frame = int(getattr(self._env.env, "final_frame", 0))
        except Exception:
            final_frame = 0
        try:
            step_frames = int(getattr(self._env.env, "step_frames", 1))
        except Exception:
            step_frames = 1
        if self._step_frames is not None:
            try:
                step_frames = int(self._step_frames)
            except Exception:
                pass

        if final_frame <= 1:
            return 0

        if bool(self._spec.allow_short_tail):
            max_start = final_frame - 1
        else:
            max_start = (final_frame - 1) - (int(self._spec.max_steps) * step_frames)
        max_start = max(0, int(max_start))

        lo = max(0, int(self._spec.start_min))
        hi = max(lo, int(max_start))
        if self._spec.start_max is not None:
            try:
                hi = min(int(hi), int(self._spec.start_max))
            except Exception:
                pass
        hi = max(lo, int(hi))

        if mode.startswith("seq"):
            stride = None
            try:
                if self._spec.start_stride is not None:
                    stride = int(self._spec.start_stride)
            except Exception:
                stride = None
            if stride is None or stride <= 0:
                stride = max(1, int(int(self._spec.max_steps) * step_frames))
            cur = int(self._scene_start_cursor.get(int(scene_id), lo))
            if cur < lo or cur > hi:
                cur = lo
            sf = cur
            nxt = cur + int(stride)
            if nxt > hi:
                nxt = lo
            self._scene_start_cursor[int(scene_id)] = int(nxt)
        else:
            seed = int(self._spec.ddp_seed) + int(self._spec.rank) * 100003 + int(scene_id) * 97 + int(self._reset_counter) * 1009 + int(self._spec.worker_id) * 17
            rng = np.random.RandomState(seed)
            sf = int(rng.randint(lo, hi + 1))

        if step_frames > 1:
            sf = (sf // step_frames) * step_frames
        return int(sf)

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        max_attempts = max(1, len(self._scene_ids))
        attempts = 0
        last_err: Exception | None = None
        while attempts < max_attempts:
            sid = self._pick_scene_id()
            try:
                sf = self._sample_start_frame(scene_id=sid)
                self._reset_counter += 1
                obs, info = self._env.reset(scene=int(sid), start_frame=int(sf), step_frames=self._step_frames)
                self._current_scene = int(sid)
                if info is None:
                    info = {}
                if isinstance(info, dict):
                    info.setdefault("scene", int(self._current_scene))
                    try:
                        info.setdefault("now_frame", getattr(self._env.env, "now_frame", None))
                    except Exception:
                        pass
                    info.setdefault("start_frame", int(sf))
                self._last_obs = obs
                self._last_info = dict(info) if isinstance(info, dict) else {}
                return obs, info
            except FileNotFoundError as e:
                last_err = e
                attempts += 1
                continue
            except Exception as e:
                last_err = e
                attempts += 1
                continue
        raise RuntimeError(f"SceneSamplingEnv.reset failed after {max_attempts} attempts: {last_err}")

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if info is None:
            info = {}
        if isinstance(info, dict):
            info.setdefault("scene", int(self._current_scene))
            try:
                info.setdefault("now_frame", getattr(self._env.env, "now_frame", None))
            except Exception:
                pass
        self._last_obs = obs
        self._last_info = dict(info) if isinstance(info, dict) else {}
        return obs, float(reward), bool(terminated), bool(truncated), info

    def idle(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Return last observation/info without stepping."""
        if self._last_obs is None:
            obs, info = self.reset()
            return obs, info
        return self._last_obs, (self._last_info or {})

    def finalize_episode_reward(self, *, done_reason: str = "timeout") -> tuple[float, Dict[str, Any]]:
        return self._env.finalize_episode_reward(done_reason=str(done_reason))


def make_scene_sampling_env(
    *,
    cuda: int,
    reward_cfg: Dict[str, Any] | None,
    debug: bool,
    scene_ids: List[int],
    scene_sampling: str,
    ddp_seed: int,
    rank: int,
    worker_id: int,
    start_mode: str,
    allow_short_tail: bool,
    start_min: int,
    start_max: Optional[int],
    start_stride: Optional[int],
    max_steps: int,
    render_w: int | None = None,
    render_h: int | None = None,
    step_frames: int | None = None,
) -> SceneSamplingEnv:
    spec = SceneSamplingSpec(
        scene_ids=list(scene_ids),
        scene_sampling=str(scene_sampling),
        ddp_seed=int(ddp_seed),
        rank=int(rank),
        worker_id=int(worker_id),
        start_mode=str(start_mode),
        allow_short_tail=bool(allow_short_tail),
        start_min=int(start_min),
        start_max=(int(start_max) if start_max is not None else None),
        start_stride=(int(start_stride) if start_stride is not None else None),
        max_steps=int(max_steps),
    )
    return SceneSamplingEnv(
        cuda=int(cuda),
        reward_cfg=reward_cfg,
        debug=bool(debug),
        spec=spec,
        render_w=(int(render_w) if render_w is not None else None),
        render_h=(int(render_h) if render_h is not None else None),
        step_frames=(int(step_frames) if step_frames is not None else None),
    )


def _worker(remote: Any, parent_remote: Any, env_fn: Callable[[], Any]) -> None:
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs, info = env.reset()
                remote.send((obs, info))
            elif cmd == "reset_one":
                obs, info = env.reset()
                remote.send((obs, info))
            elif cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                remote.send((obs, reward, terminated, truncated, info))
            elif cmd == "idle":
                obs, info = env.idle()
                remote.send((obs, 0.0, False, False, info))
            elif cmd == "call":
                name, args, kwargs = data
                res = getattr(env, name)(*args, **kwargs)
                remote.send(res)
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown command: {cmd}")
    except EOFError:
        # Main process died.
        try:
            remote.close()
        except Exception:
            pass


class SubprocVecEnv:
    """Minimal SubprocVecEnv for non-gym environments.

    - Each worker runs one env instance.
    - Observation/info are assumed to be picklable (dict of numpy arrays is OK).
    - reset() resets all envs; reset_one(i) resets a single env.

    This is intentionally small and tailored for RLReconEnv-style API.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], Any]],
        *,
        start_method: str = "spawn",
    ) -> None:
        if len(env_fns) <= 0:
            raise ValueError("SubprocVecEnv requires at least one env_fn")

        ctx = mp.get_context(start_method)
        self._closed = False
        self._n = int(len(env_fns))

        self._remotes, self._work_remotes = zip(*[ctx.Pipe() for _ in range(self._n)])
        self._ps: List[mp.Process] = []
        for wr, r, fn in zip(self._work_remotes, self._remotes, env_fns):
            p = ctx.Process(target=_worker, args=(wr, r, fn), daemon=True)
            p.start()
            wr.close()
            self._ps.append(p)

    @property
    def num_envs(self) -> int:
        return self._n

    def reset(self) -> Tuple[List[Any], List[Dict[str, Any]]]:
        self._assert_not_closed()
        for r in self._remotes:
            r.send(("reset", None))
        results = [r.recv() for r in self._remotes]
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def reset_one(self, i: int) -> Tuple[Any, Dict[str, Any]]:
        self._assert_not_closed()
        idx = int(i)
        self._remotes[idx].send(("reset_one", None))
        obs, info = self._remotes[idx].recv()
        return obs, info

    def step(self, actions: List[Any]) -> Tuple[List[Any], List[float], List[bool], List[bool], List[Dict[str, Any]]]:
        self._assert_not_closed()
        if len(actions) != self._n:
            raise ValueError(f"actions length {len(actions)} != num_envs {self._n}")
        for r, a in zip(self._remotes, actions):
            if a is None:
                r.send(("idle", None))
            else:
                r.send(("step", a))
        results = [r.recv() for r in self._remotes]
        obs, rew, term, trunc, info = zip(*results)
        return list(obs), list(rew), list(term), list(trunc), list(info)

    def call(self, name: str, *args: Any, **kwargs: Any) -> List[Any]:
        self._assert_not_closed()
        for r in self._remotes:
            r.send(("call", (name, args, kwargs)))
        return [r.recv() for r in self._remotes]

    def call_one(self, i: int, name: str, *args: Any, **kwargs: Any) -> Any:
        self._assert_not_closed()
        idx = int(i)
        self._remotes[idx].send(("call", (name, args, kwargs)))
        return self._remotes[idx].recv()

    def get_attr(self, name: str) -> List[Any]:
        self._assert_not_closed()
        for r in self._remotes:
            r.send(("get_attr", name))
        return [r.recv() for r in self._remotes]

    def close(self) -> None:
        if self._closed:
            return
        for r in self._remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self._ps:
            try:
                p.join(timeout=2.0)
            except Exception:
                pass
        self._closed = True

    def _assert_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("SubprocVecEnv is closed")


class SerialVecEnv:
    """Single-process vectorized env runner.

    This is a drop-in replacement for `SubprocVecEnv` with the same public API,
    but it runs all env instances in the current process (no multiprocessing).

    Why this exists:
    - GPU-heavy environments (CUDA init + neural rendering) often deadlock or
      become extremely fragile under `spawn`.
    - Keeping everything in-process avoids CUDA context re-init and zombie
      workers, while still enabling batched policy inference across multiple envs.
    """

    def __init__(self, env_fns: List[Callable[[], Any]]) -> None:
        if len(env_fns) <= 0:
            raise ValueError("SerialVecEnv requires at least one env_fn")
        self._closed = False
        self._envs = [fn() for fn in env_fns]

    @property
    def num_envs(self) -> int:
        return int(len(self._envs))

    def reset(self) -> Tuple[List[Any], List[Dict[str, Any]]]:
        self._assert_not_closed()
        results = [e.reset() for e in self._envs]
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def reset_one(self, i: int) -> Tuple[Any, Dict[str, Any]]:
        self._assert_not_closed()
        idx = int(i)
        obs, info = self._envs[idx].reset()
        return obs, info

    def step(
        self, actions: List[Any]
    ) -> Tuple[List[Any], List[float], List[bool], List[bool], List[Dict[str, Any]]]:
        self._assert_not_closed()
        if len(actions) != self.num_envs:
            raise ValueError(f"actions length {len(actions)} != num_envs {self.num_envs}")

        obs_list: List[Any] = []
        rew_list: List[float] = []
        term_list: List[bool] = []
        trunc_list: List[bool] = []
        info_list: List[Dict[str, Any]] = []

        for env, a in zip(self._envs, actions):
            if a is None:
                # Keep interface identical to SubprocVecEnv: None means "idle".
                if hasattr(env, "idle"):
                    obs, info = env.idle()
                else:
                    # Fallback: do not advance; best-effort reset.
                    obs, info = env.reset()
                obs_list.append(obs)
                rew_list.append(0.0)
                term_list.append(False)
                trunc_list.append(False)
                info_list.append(info if isinstance(info, dict) else {})
            else:
                obs, rew, term, trunc, info = env.step(a)
                obs_list.append(obs)
                rew_list.append(float(rew))
                term_list.append(bool(term))
                trunc_list.append(bool(trunc))
                info_list.append(info if isinstance(info, dict) else {})

        return obs_list, rew_list, term_list, trunc_list, info_list

    def call(self, name: str, *args: Any, **kwargs: Any) -> List[Any]:
        self._assert_not_closed()
        return [getattr(e, name)(*args, **kwargs) for e in self._envs]

    def call_one(self, i: int, name: str, *args: Any, **kwargs: Any) -> Any:
        self._assert_not_closed()
        idx = int(i)
        return getattr(self._envs[idx], name)(*args, **kwargs)

    def get_attr(self, name: str) -> List[Any]:
        self._assert_not_closed()
        return [getattr(e, name) for e in self._envs]

    def close(self) -> None:
        if self._closed:
            return
        for e in self._envs:
            try:
                if hasattr(e, "close"):
                    e.close()
            except Exception:
                pass
        self._closed = True

    def _assert_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("SerialVecEnv is closed")
