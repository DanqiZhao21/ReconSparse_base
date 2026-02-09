import numpy as np

import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# plan_anchors 文件路径
file_path = os.path.join(_REPO_ROOT, "assets", "nus", "anchor", "traj_anchor_05s_3721.npy")

# 尝试加载文件
try:
    anchors = np.load(file_path, allow_pickle=True)
except Exception as e:
    print(f"Failed to load file: {e}")
    raise

# 打印 anchors 类型和整体 shape
print("anchors type:", type(anchors))
print("anchors shape:", anchors.shape if hasattr(anchors, 'shape') else 'N/A')

# 打印第一个元素的信息
first_elem = anchors[0]
first_elem_shape = first_elem.shape if hasattr(first_elem, 'shape') else 'N/A'
print("first element type:", type(first_elem))
print("first element shape:", first_elem_shape)
print("first element content:\n", first_elem)
