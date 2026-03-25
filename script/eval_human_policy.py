import os
import io
import sys
current_dir = os.getcwd()
sys.path.append(current_dir)
import time
import numpy as np
import requests
import imageio
import gymnasium as gym
from gymnasium.envs.registration import register
from reconsimulator.envs.nus import ReconSimulator
# --------------------------- Configuration --------------------------- #
SERVER_URL = "http://127.0.0.1:8001/get_action/file"
SAVE_DIR = "./visual/human"
os.makedirs(SAVE_DIR, exist_ok=True)

register(
    id="ReconSimulator-v0",
    entry_point="reconsimulator.envs.nus:ReconSimulator",
)


# --------------------------- Helper Functions ------------------------ #
def send_observation_until_success_np(observation_dict: dict) -> dict:#NOTE 输入：observation_dict;输出：服务器返回的 JSON 解析后的字典:{"action": [ax_index, ay_index, flag]}
    while True:
        try:
            # Validate that observation_dict contains numpy arrays
            if not observation_dict or not all(isinstance(v, np.ndarray) for v in observation_dict.values()):
                print("[WARN] Observation data is empty or invalid, skipping send.")
                return {}

            # Save all arrays to a compressed npz in memory
            buf = io.BytesIO()
            np.savez_compressed(buf, **observation_dict)
            buf.seek(0)

            # Multipart/form-data upload
            files = {
                "obs": ("obs.npz", buf, "application/octet-stream")
            }

            response = requests.post(SERVER_URL, files=files, timeout=30)

            if response.status_code == 200:
                try:
                    return response.json()
                except Exception as e:
                    print(f"[ERROR] Failed to parse server JSON response: {e}")
                    print("Response content:", response.text[:500])
                    time.sleep(1)
            else:
                print(f"[ERROR] Server returned {response.status_code}: {response.text[:200]}")
                time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"[WARN] Request exception: {e}, retrying in 1 second...")
            time.sleep(1)
        finally:
            buf.close()


# --------------------------- Main Environment Loop ------------------- #
def test_gridworld_env():
    """
    Run the ReconSimulator environment, sending observations to the server
    and receiving actions. Saves each scene as an .mp4 video.
    """
    env = gym.make("ReconSimulator-v0", cuda=0, scene=0, debug=False)

    for scene in range(0, 400):
        print(f"\n=== Running Scene {scene} ===")
        try:
            obs, info = env.reset(seed=scene, options=None)
            terminated, truncated = False, False
            demo_video = []

            while not terminated and not truncated:
                # Assemble observation dictionary
                observation = {
                    "back_left": obs["back_left"],
                    "front_left": obs["front_left"],
                    "front": obs["front"],
                    "front_right": obs["front_right"],
                    "back_right": obs["back_right"],
                    "back": obs["back"],
                }

                # Get action from server
                action_data = send_observation_until_success_np(observation)
                if not action_data or "action" not in action_data:
                    print(f"[WARN] Scene {scene}: No valid action received, breaking loop.")
                    break

                action = action_data["action"]

                # Convert action to list of numpy arrays if needed
                try:
                    action_for_env = [np.array(a) for a in action]
                except Exception:
                    action_for_env = action

                # Step environment
                step_out = env.step(action_for_env)
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, info = step_out
                else:
                    obs, terminated, truncated, info = step_out

                # Collect video frames
                final_image = np.hstack([
                    obs["back_left"],
                    obs["front_left"],
                    obs["front"],
                    obs["front_right"],
                    obs["back_right"],
                ])
                demo_video.append(final_image)

            # Save video
            if demo_video:
                out_path = os.path.join(SAVE_DIR, f"{scene}.mp4")
                imageio.mimwrite(out_path, demo_video, fps=10)
                print(f"[INFO] Scene {scene} saved video: {out_path} ({len(demo_video)} frames)")
            else:
                print(f"[WARN] Scene {scene} has no frames, skipping video save.")

        except Exception as e:
            print(f"[ERROR] Scene {scene} execution failed: {e}")
            time.sleep(1)
            continue


# --------------------------- Entrypoint ------------------------------ #
if __name__ == "__main__":
    print("[CLIENT] Starting numpy transfer test client...")
    test_gridworld_env()
