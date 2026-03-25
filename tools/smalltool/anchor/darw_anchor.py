
#NOTE
# =========================================
# arr shape: (3721, 6, 2)
# first element shape: (2,)
# first element: [0. 0.]
# =============================================
# import numpy as np
# path = "$REPO_ROOT/assets/nus/anchor/traj_anchor_05s_3721.npy"
# arr = np.load(path, allow_pickle=True)
# print("arr shape:", arr.shape)
# print("first element shape:", arr[0, 0].shape)
# print("first element:", arr[0, 0])

#NOTE
# =========================================
# arr shape: (3721, 1)
# first element shape: ()
# first element: False
# =============================================
# import numpy as np
# path = "$REPO_ROOT/assets/nus/anchor/traj_anchor_05s_3721_mask.npy"
# arr = np.load(path, allow_pickle=True)
# print("arr shape:", arr.shape)
# print("first element shape:", arr[0, 0].shape)
# print("first element:", arr[0, 0])

#NOTE ALL TRAJ IN BLUE
# import numpy as np
# import matplotlib.pyplot as plt

# # === 路径 ===
# path = "$REPO_ROOT/assets/nus/anchor/traj_anchor_05s_3721.npy"

# # === 加载数据 ===
# traj = np.load(path)   # (3721, 6, 2)
# print("traj shape:", traj.shape)

# num_traj, T, _ = traj.shape

# # === 开始画图 ===
# plt.figure(figsize=(8, 8))

# for i in range(num_traj):
#     xy = traj[i]            # (6, 2)
#     plt.plot(
#         xy[:, 0],
#         xy[:, 1],
#         color="blue",
#         alpha=0.05,
#         linewidth=0.8
#     )

# # 画 ego 原点
# plt.scatter(0, 0, c="red", s=40, label="ego")

# plt.axis("equal")
# plt.grid(True, linestyle="--", alpha=0.4)
# plt.xlabel("x (m)")
# plt.ylabel("y (m)")
# plt.title("Trajectory Anchors (3721 x 6)")
# plt.legend()
# plt.tight_layout()

# out_path = "traj_anchor_05s_3721.png"
# plt.savefig(out_path, dpi=200, bbox_inches="tight")


#NOTE SAMPLE 200 TRAJ
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================
# 配置
# ===============================
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
path = os.path.join(_REPO_ROOT, "assets", "nus", "anchor", "traj_anchor_05s_3721.npy")
num_sample = 200
seed = 0  # 固定随机种子，方便复现（可删）

# ===============================
# 加载数据
# ===============================
traj = np.load(path)   # (3721, 6, 2)
print("traj shape:", traj.shape)

num_traj, T, _ = traj.shape

# ===============================
# 随机抽样轨迹索引
# ===============================
np.random.seed(seed)
idx = np.random.choice(num_traj, size=num_sample, replace=False)

# ===============================
# 绘制
# ===============================
plt.figure(figsize=(8, 8))

for i in idx:
    xy = traj[i]   # (6, 2)

    # 画轨迹线
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        color="blue",
        alpha=0.4,
        linewidth=1.2
    )

    # 画每个时间步的点
    plt.scatter(
        xy[:, 0],
        xy[:, 1],
        color="red",
        s=12,
        alpha=0.8
    )

# ego 原点
plt.scatter(0, 0, c="black", s=60, marker="x", label="ego")

plt.axis("equal")
plt.grid(True, linestyle="--", alpha=0.4)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Random 200 Trajectory Anchors (with timestep markers)")
plt.legend()
plt.tight_layout()
out_path = "traj_anchor_05s_3721.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
