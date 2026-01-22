import os
import numpy as np
from scipy.spatial import cKDTree

# 使用项目里的默认路径（nus_config.py 指向）
DEFAULT_PATH = "/root/clone/ReconDreamer-RL/assets/nus/anchor/traj_anchor_05s_3721.npy"

def main(path: str = DEFAULT_PATH):
    anchors = np.load(path)
    # anchors: (N, T, 2) 其中 2 是 (x, y) 米
    N, T, _ = anchors.shape

    # 时间尺度：文件名含 05s → 0.5s 间隔；总时长 = (T-1)*0.5s
    # dt = 0.
    # horizon_s = (T - 1) * dt
    horizon_s = 0.5
    dt=horizon_s/(T-1)
    # 末帧分布范围（覆盖半径/矩形范围）
    last = anchors[:, -1, :]  # (N, 2)
    min_x, max_x = float(last[:, 0].min()), float(last[:, 0].max())
    min_y, max_y = float(last[:, 1].min()), float(last[:, 1].max())

    # 最近邻典型间距（末帧）
    tree = cKDTree(last)
    # 查询每个点最近的两个邻居（含自身 → 跳过 index=0 自身距离）
    dists, idxs = tree.query(last, k=2)
    nn = dists[:, 1]  # 最近的非自身邻居距离
    nn_median = float(np.median(nn))
    nn_mean = float(np.mean(nn))

    # 每个时间步的位移（米）统计
    step_disp = np.linalg.norm(anchors[:, 1:, :] - anchors[:, :-1, :], axis=-1)  # (N, T-1)
    step_median = [float(np.median(step_disp[:, i])) for i in range(T - 1)]
    step_mean = [float(np.mean(step_disp[:, i])) for i in range(T - 1)]

    print("=== Anchor Stats ===")
    print(f"path            : {path}")
    print(f"shape           : {anchors.shape}  # (N, T, 2), 单位: 米")
    print(f"time step (dt)  : {dt} s")
    print(f"horizon         : {horizon_s} s  # {(T-1)} 步 × {dt}s")
    print(f"last-x range    : [{min_x:.3f}, {max_x:.3f}] m")
    print(f"last-y range    : [{min_y:.3f}, {max_y:.3f}] m")
    print(f"NN spacing (med): {nn_median:.3f} m; mean: {nn_mean:.3f} m  # 末帧最近邻典型间距")
    print("per-step disp median (m):", [round(v, 3) for v in step_median])
    print("per-step disp mean   (m):", [round(v, 3) for v in step_mean])

if __name__ == "__main__":
    p = os.environ.get("PLAN_ANCHOR_PATH", DEFAULT_PATH)
    main(p)
