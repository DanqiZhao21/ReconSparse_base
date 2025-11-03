# nus_config.py
import os

 
DATA_ROOT = "assets/nus"
BASE_DATA_DIR = os.path.join(DATA_ROOT, "data")
INFO_DIR = os.path.join(DATA_ROOT, "others")

ALL_CAMS_FILE   = os.path.join(DATA_ROOT, "others", "all_cams.pkl")
ALL_IMAGES_FILE = os.path.join(DATA_ROOT, "others", "all_images.pkl")


PLAN_ANCHORS_FILE      = os.path.join(DATA_ROOT, "anchor", "traj_anchor_05s_3721.npy")
PLAN_ANCHORS_YAW_FILE  = os.path.join(DATA_ROOT, "anchor", "traj_anchor_05s_3721_yaw.npy")
PLAN_ANCHORS_MASK_FILE = os.path.join(DATA_ROOT, "anchor", "traj_anchor_05s_3721_mask.npy")


FRAME2TOKEN_DIR = os.path.join(DATA_ROOT,"information", "frame2token") 
TOKEN2VAD_FILE  = os.path.join(DATA_ROOT, "information", "token2vad.pkl")