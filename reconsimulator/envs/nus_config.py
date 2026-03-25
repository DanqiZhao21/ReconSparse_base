# nus_config.py
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_ROOT = os.path.join(_REPO_ROOT, "assets", "nus")
BASE_DATA_DIR = os.path.join(DATA_ROOT, "data")
INFO_DIR = os.path.join(DATA_ROOT, "others")

ALL_CAMS_FILE   = os.path.join(DATA_ROOT, "others", "all_cams.pkl")
ALL_IMAGES_FILE = os.path.join(DATA_ROOT, "others", "all_images.pkl")


PLAN_ANCHORS_FILE      = os.path.join(DATA_ROOT, "anchor", "traj_anchor_05s_3721.npy")
PLAN_ANCHORS_YAW_FILE  = os.path.join(DATA_ROOT, "anchor", "traj_anchor_05s_3721_yaw.npy")
PLAN_ANCHORS_MASK_FILE = os.path.join(DATA_ROOT, "anchor", "traj_anchor_05s_3721_mask.npy")


FRAME2TOKEN_DIR = os.path.join(DATA_ROOT,"information", "frame2token") 
TOKEN2VAD_FILE  = os.path.join(DATA_ROOT, "information", "token2vad.pkl")

# Optional: full nuScenes dataset root (for official can_bus access).
# If present, ReconSimulator can initialize ego vel/acc from NuScenesCanBus.
NUSCENES_DATA_ROOT = os.environ.get(
	"NUSCENES_DATA_ROOT",
	os.path.join(_REPO_ROOT, "assets", "nuscenes"),
)
NUSCENES_VERSION = os.environ.get("NUSCENES_VERSION", "v1.0-trainval")