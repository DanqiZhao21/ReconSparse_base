import os
import copy
import math
import json
import pickle
import torch
import numpy as np
import gymnasium as gym
from reconsimulator.envs.tool import get_splat,get_sky_view,move_to_device,slerp
from reconsimulator.envs import nus_config as cfg
from scipy.spatial.transform import Slerp, Rotation as R
from scipy.spatial.distance import cdist

TRANSFORM_MATRIX = torch.eye(4, dtype=torch.float32).cuda()

class ReconSimulator(gym.Env):
    def __init__(self, cuda=0, scene=0, debug=True):
        self.device = f"cuda:{cuda}"
        self.debug = debug
        self.scene = scene
        self.w, self.h = 800, 450

        # Observation space: 6 camera RGB views
        self.observation_space = gym.spaces.Dict({
            name: gym.spaces.Box(low=0, high=255, shape=(self.h, self.w, 3), dtype=np.uint8)
            for name in ["front", "front_left", "front_right", "back_left", "back_right", "back"]
        })

        # Action space: discrete anchor indices
        self.action_space = gym.spaces.MultiDiscrete([61, 61])

        # Load trainer
        self.trainer, self.num_timesteps = get_splat(self.device, self.scene)
        self.trainer.eval()

        # Frame control
        self.step_frames = 1
        self.final_frame = 186
        self.now_frame = 0

        # Load all data
        self._load_camera_and_images()
        self._load_ego_and_cam_matrices()
        self._load_expert_ego_frames()
        self._load_plan_anchors()
        self._load_token_mappings()

        self.all_camera_now = []
        self.get_all_point_for_expert()

    # ------------------------- Private loading functions ------------------------ #
    # ALL_CAMS_FILE   = os.path.join(DATA_ROOT, "others", "all_cams.pkl") 6 个相机的「静态相机参数模板」
    # ALL_IMAGES_FILE = os.path.join(DATA_ROOT, "others", "all_images.pkl")
    def _load_camera_and_images(self):
        with open(cfg.ALL_CAMS_FILE, "rb") as f:
            self.all_cams = pickle.load(f)
        with open(cfg.ALL_IMAGES_FILE, "rb") as f:
            self.all_images = pickle.load(f)

    def _load_ego_and_cam_matrices(self):
        cam2ego = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/cam2ego/0.txt"))
        ego2world = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/000.txt"))
        self.camera_front_start = ego2world @ cam2ego
        self.start_ego = np.linalg.inv(self.camera_front_start) @ ego2world

        # Load all camera-to-ego matrices
        self.cam2ego = [
            np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/cam2ego/{i}.txt"))
            for i in range(6)
            if os.path.exists(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/cam2ego/{i}.txt"))
        ]

    def _load_expert_ego_frames(self):#NOTE 专家车辆轨迹（ground-truth trajectory）:世界坐标到前置相机起始坐标的相对变换
        self.all_expert_ego = []
        for i in range(0, self.final_frame + self.step_frames, self.step_frames):
            expert_world = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/{i:03d}.txt"))
            expert_world = np.linalg.inv(self.camera_front_start) @ expert_world
            self.all_expert_ego.append(expert_world)

    def _load_plan_anchors(self):
        self.plan_anchors = torch.from_numpy(np.load(cfg.PLAN_ANCHORS_FILE).astype(np.float32))
        self.plan_anchors_yaw = torch.from_numpy(np.load(cfg.PLAN_ANCHORS_YAW_FILE).astype(np.float32)) * 5
        self.plan_anchors_mask = torch.from_numpy(np.load(cfg.PLAN_ANCHORS_MASK_FILE).reshape(-1))
        self.x_anchor = 61
        self.y_anchor = 61

    def _load_token_mappings(self):
        frame2token_path = os.path.join(cfg.FRAME2TOKEN_DIR, f"{self.scene:03d}.json")
        with open(frame2token_path, 'r') as f:
            data = json.load(f)
            self.frame2token = {v: k for k, v in data.items()}
        with open(cfg.TOKEN2VAD_FILE, 'rb') as f:
            self.token2vad = pickle.load(f)

    # ------------------------- Observation & Info ------------------------ #
    def _get_obs(self):
        """
        Compute observation images from all active cameras using the trainer.根据当前相机信息生成可观察的 RGB 图像
        """
        self.now_observe_image = []
        with torch.no_grad():
            for cam in self.all_camera_now:
                cam_info, img_info = cam
                results = self.trainer(img_info, cam_info)#NOTE self.trainer(img_info, cam_info)：
                rgb = results['rgb'].clamp(0, 1).cpu().numpy()
                scaled_rgb = (rgb * 255).astype(np.uint8)
                self.now_observe_image.append(scaled_rgb)
        self.all_camera_now = []
        return {
            "front": self.now_observe_image[0],
            "front_left": self.now_observe_image[1],
            "front_right": self.now_observe_image[2],
            "back_left": self.now_observe_image[3],
            "back_right": self.now_observe_image[4],
            "back": self.now_observe_image[5],
        }

    def _get_info(self):
        return {
            "exp_pos": getattr(self, "last_exp_pos", None),
            "act_pos": getattr(self, "last_act_pos", None),
            "exp_yaw_deg": getattr(self, "last_exp_yaw_deg", None),
            "act_yaw_deg": getattr(self, "last_act_yaw_deg", None),
            "xz_err_m": getattr(self, "last_xz_err_m", None),
            "yaw_err_deg": getattr(self, "last_yaw_err_deg", None),
        }

    # ------------------------- Gym API ------------------------ #
    def reset(self, seed=None, options=None):#NOTE 重置环境，重新开始一个新场景。
        self.update(seed)
        self.start_ego = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/000.txt"))
        self.start_ego = np.linalg.inv(self.camera_front_start) @ self.start_ego

        self.all_camera_now = []
        for i in range(6):
            cam_info = copy.deepcopy(self.all_cams[i])
            cam_info = move_to_device(cam_info, self.device)
            cam_info['camera_to_world'] = torch.tensor(self.start_ego @ self.cam2ego[i], device=self.device, dtype=torch.float32)
            cam_info['camera_to_world'] = cam_info['camera_to_world'] @ TRANSFORM_MATRIX

            img_info = copy.deepcopy(self.all_images[i])
            img_info = move_to_device(img_info, self.device)
            img_info['origins'], img_info['viewdirs'], img_info['direction_norm'] = get_sky_view(
                cam_info['camera_to_world'], cam_info['intrinsics'], self.device, self.h, self.w
            )
            img_info['normed_time'] = torch.tensor(self.trainer.normalized_timestamps[self.now_frame].item())
            self.all_camera_now.append((cam_info, img_info))

        return self._get_obs(), self._get_info()
#ADD 修改了一下step函数的逻辑
    
    def step(self, action):#NOTE 根据动作 action 更新车辆状态（ego pose
        self.now_frame += self.step_frames
        ax_index, ay_index, flag = action

        # --- 计算专家下一帧位姿（world→front-start 相对变换） ---
        expert_next_ego = np.linalg.inv(self.camera_front_start) @ np.loadtxt(
            os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/{self.now_frame:03d}.txt")
        )

        # --- 计算 action 锚点推进的“假设下一帧”位姿（用于对比或真实推进） ---
        selected_idx = ax_index * self.y_anchor + ay_index
        future_xy = self.plan_anchors[selected_idx][-1, :]
        if torch.is_tensor(future_xy):
            future_x = float(future_xy[0].item())
            future_y = float(future_xy[1].item())
        else:
            future_x = float(future_xy[0])
            future_y = float(future_xy[1])

        future_yaw_v = self.plan_anchors_yaw[selected_idx]
        if torch.is_tensor(future_yaw_v):
            future_yaw = float(future_yaw_v.item())
        else:
            future_yaw = float(future_yaw_v)

        tpt = np.array([
            [math.cos(future_yaw), -math.sin(future_yaw), 0, future_x],
            [math.sin(future_yaw),  math.cos(future_yaw), 0, future_y],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        action_next_ego = self.start_ego @ tpt
        # 记录用于 info 的位置（x,y,z）与航向/误差
        self.last_exp_pos = expert_next_ego[:3, 3].copy()
        self.last_act_pos = action_next_ego[:3, 3].copy()
        def _yaw_from_R(Rm):
            return math.atan2(Rm[1, 0], Rm[0, 0])
        exp_yaw = _yaw_from_R(expert_next_ego[:3, :3])
        act_yaw = _yaw_from_R(action_next_ego[:3, :3])
        pos_delta = self.last_act_pos - self.last_exp_pos
        self.last_xz_err_m = float(np.linalg.norm(pos_delta[[0, 2]]))
        self.last_yaw_err_deg = abs((act_yaw - exp_yaw) * 180.0 / math.pi)
        self.last_exp_yaw_deg = float(exp_yaw * 180.0 / math.pi)
        self.last_act_yaw_deg = float(act_yaw * 180.0 / math.pi)

        # --- 根据 debug/flag 选择真实推进 ---
        print(f"self.debug is {self.debug}, flag is {flag}")
        if self.debug:
            # Debug 模式：使用专家推进，并打印与 action 推进的差异
            self.start_ego = expert_next_ego
            self.start_ego[1][-1] = self.updateGroundDistance()

            exp_pos = self.last_exp_pos
            act_pos = self.last_act_pos
            pos_delta = act_pos - exp_pos
            pos_xz_err = self.last_xz_err_m
            exp_yaw = self.last_exp_yaw_deg
            act_yaw = self.last_act_yaw_deg
            yaw_err_deg = self.last_yaw_err_deg

            print(
                f"🐅[Frame {self.now_frame:03d}] action(ax={ax_index}, ay={ay_index}, flag={flag}) | "
                f"expert_pos=({exp_pos[0]:.3f},{exp_pos[1]:.3f},{exp_pos[2]:.3f}) yaw={exp_yaw:.2f}deg "
                f"action_pos=({act_pos[0]:.3f},{act_pos[1]:.3f},{act_pos[2]:.3f}) yaw={act_yaw:.2f}deg "
                f"delta=({pos_delta[0]:.3f},{pos_delta[1]:.3f},{pos_delta[2]:.3f}); "
                f"xz_err={pos_xz_err:.3f}m, yaw_err={yaw_err_deg:.2f}deg"
            )
        else:
            # 非 debug 模式：遵循原始逻辑（flag=1 走专家；否则走 action）
            if flag:
                self.start_ego = expert_next_ego
            else:
                self.start_ego = action_next_ego
            
            exp_pos = self.last_exp_pos
            act_pos = self.last_act_pos
            pos_delta = act_pos - exp_pos
            pos_xz_err = self.last_xz_err_m
            exp_yaw = self.last_exp_yaw_deg
            act_yaw = self.last_act_yaw_deg
            yaw_err_deg = self.last_yaw_err_deg

            print(
                f"🐅[Frame {self.now_frame:03d}] action(ax={ax_index}, ay={ay_index}, flag={flag}) | "
                f"expert_pos=({exp_pos[0]:.3f},{exp_pos[1]:.3f},{exp_pos[2]:.3f}) yaw={exp_yaw:.2f}deg "
                f"action_pos=({act_pos[0]:.3f},{act_pos[1]:.3f},{act_pos[2]:.3f}) yaw={act_yaw:.2f}deg "
                f"delta=({pos_delta[0]:.3f},{pos_delta[1]:.3f},{pos_delta[2]:.3f}); "
                f"xz_err={pos_xz_err:.3f}m, yaw_err={yaw_err_deg:.2f}deg"
            )
            self.start_ego[1][-1] = self.updateGroundDistance()
            
#ADD
        w,h = 800,450
        for i in range(6):#NOTE 更新相机信息
            loaded_cam_infos = copy.deepcopy(self.all_cams[i])
            loaded_cam_infos = move_to_device(loaded_cam_infos,self.device)
            loaded_cam_infos['camera_to_world'] = torch.tensor(self.start_ego @ self.cam2ego[i]).to(self.device).to(torch.float32)
            loaded_cam_infos['camera_to_world'] = loaded_cam_infos['camera_to_world'] @ TRANSFORM_MATRIX
            loaded_img_infos = copy.deepcopy(self.all_images[i])
            loaded_img_infos = move_to_device(loaded_img_infos,self.device)
            loaded_img_infos['origins'],\
            loaded_img_infos['viewdirs'], \
            loaded_img_infos['direction_norm'] = get_sky_view(loaded_cam_infos['camera_to_world'],\
                                                                  loaded_cam_infos['intrinsics'],\
                                                                    self.device,h,w)
            loaded_img_infos['normed_time'] = torch.tensor(self.trainer.normalized_timestamps[self.now_frame].item()) 
            self.all_camera_now.append((loaded_cam_infos,loaded_img_infos))
        observation = self._get_obs()

        terminated, truncated = False, False
        if self.now_frame == self.final_frame - 1:
            terminated = True
        else:
            terminated = False
        
        if self.check_coliision == True:
            truncated = True
        else:
            truncated = False
        
        return observation, terminated, truncated, self._get_info()
    
    def check_coliision(self):
        return False
    
    def update(self, scene: int, *, step_frames: int = None):
        self.scene = int(scene)
        if step_frames is not None:
            self.step_frames = int(step_frames)        
        self.now_frame = 0
        self.all_camera_now = []
        self.save = None

        self.trainer, self.num_timesteps = get_splat(self.device, self.scene)
        self.trainer.eval()

        with open(cfg.ALL_CAMS_FILE, "rb") as f:
            self.all_cams = pickle.load(f)
        with open(cfg.ALL_IMAGES_FILE, "rb") as f:
            self.all_images = pickle.load(f)

        cam2ego_0 = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/cam2ego/0.txt"))
        ego2world_0 = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/000.txt"))
        self.camera_front_start = ego2world_0 @ cam2ego_0

        self.start_ego = np.linalg.inv(self.camera_front_start) @ ego2world_0

        self.cam2ego = []
        for i in range(6):
            cam_path = os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/cam2ego/{i}.txt")
            if os.path.exists(cam_path):
                self.cam2ego.append(np.loadtxt(cam_path))

        self.all_expert_ego = []
        for i in range(0, self.final_frame + self.step_frames, self.step_frames):
            expert_world = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/{i:03d}.txt"))
            expert_world = np.linalg.inv(self.camera_front_start) @ expert_world
            self.all_expert_ego.append(expert_world)
        self.get_all_point_for_expert()

    def get_all_point_for_expert(self):
        self.expert_world_all = []
        for i in range(len(self.all_expert_ego) - 1):
            start_matrix = self.all_expert_ego[i]
            end_matrix = self.all_expert_ego[i + 1]
            for alpha in np.linspace(0, 1, 40): 
                translation = (1 - alpha) * start_matrix[:3, 3] + alpha * end_matrix[:3, 3]
                start_rot = R.from_matrix(start_matrix[:3, :3])
                end_rot = R.from_matrix(end_matrix[:3, :3])
                interp_rot = slerp(start_rot, end_rot, alpha)
                new_matrix = np.eye(4)
                new_matrix[:3, :3] = interp_rot.as_matrix()
                new_matrix[:3, 3] = translation
                self.expert_world_all.append(new_matrix)

        self.expert_pair = [matrix[:3, 3][[0, 2]] for matrix in self.expert_world_all]
        self.expert_altitude  = [matrix[:3, 3][[1]] for matrix in self.expert_world_all]


    def updateGroundDistance(self):#NOTE 用当前 x,z 找到离自己最近的 expert 点;取这个 expert 点的 y 作为地面高度
        start_ego_position = self.start_ego[:3, 3][[0, 2]]
        distances = cdist([start_ego_position], self.expert_pair, 'euclidean')[0]
        nearest_indices = np.argsort(distances)[:1] 
        # print(nearest_indices)
        return self.expert_altitude[nearest_indices[0]]