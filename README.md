**ReconDreamer-RL + DiffusionDriveV2 简版使用说明**

- **目标:** 统一使用本仓库，快速完成环境配置与数据下载；以 [environment.yml](environment.yml) 为准搭建统一环境（CUDA 11.8 + Python 3.10）。
- **内容:** 克隆与环境、数据与模型下载、快速运行（缓存与评估）、常见环境变量与提示。

**前置要求**
- NVIDIA GPU（建议 CUDA 11.8），已安装驱动
- Conda（Miniconda/Anaconda）与 Git

**获取代码**
- 克隆仓库并进入目录：

```bash
git clone https://github.com/GigaAI-research/ReconDreamer-RL.git
cd ReconDreamer-RL
```

**配置环境（按 environment.yml）**
- 用根目录的环境文件创建并激活：

```bash
conda env create -f environment.yml
conda activate recondreamerNew-rl
```

- 推荐设置 `PYTHONPATH` 以确保包可见：

```bash
export PYTHONPATH=$(pwd):$(pwd)/DiffusionDriveV2:$(pwd)/DiffusionDriveV2/navsim:$PYTHONPATH
```

- 可选：以可编辑方式注册 DiffusionDriveV2（便于开发调试）：

```bash
pip install -e DiffusionDriveV2
```

**数据与模型下载（DiffusionDriveV2）**
- 进入下载脚本目录：

```bash
cd DiffusionDriveV2/download
```

- 选择需要的数据分割（示例）：
- 训练+验证：

```bash
bash download_trainval.sh
```

- 测试集：

```bash
bash download_test.sh
```

- 迷你版（快速体验）：

```bash
bash download_mini.sh
```

- 加速下载（并行）：

```bash
bash super_download.sh
```

- 可选：下载 GTRS 轨迹增强（训练模式选择器时使用）：

```bash
cd ../gtrs_traj
wget https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/navtrain_16384.pkl
```

- 模型权重：从官方资源下载并放置到 [DiffusionDriveV2/ckpt](DiffusionDriveV2/ckpt)
- `resnet34.a1_in1k`
- `diffusiondrive_navsim_88p1_PDMS`
- `diffusiondrivev2_rl.ckpt`
- `diffusiondrivev2_sel.ckpt`

**快速运行**
- 建议先进行指标与特征缓存（加速训练与评估）：

```bash
# 回到仓库根目录
cd ../../

# 设置实验输出目录
export NAVSIM_EXP_ROOT=$(pwd)/DiffusionDriveV2/exp
mkdir -p "$NAVSIM_EXP_ROOT"

# 缓存评估指标（PDMS）
python DiffusionDriveV2/navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  cache.cache_path="$NAVSIM_EXP_ROOT/metric_cache"

# 可选：缓存训练所需特征
python DiffusionDriveV2/navsim/planning/script/run_dataset_caching.py \
  agent=diffusiondrivev2_rl_agent \
  experiment_name=diffusiondrivev2_cache \
  train_test_split=navtrain
```

- 评估（快速）：

```bash
python DiffusionDriveV2/navsim/planning/script/run_pdm_score_fast.py \
  agent=diffusiondrivev2_sel_agent \
  experiment_name=diffusiondrivev2_agent_eval \
  train_test_split=navtest \
  agent.checkpoint_path=DiffusionDriveV2/ckpt/diffusiondrivev2_sel.ckpt \
  +metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache/"
```

- 也可使用本仓库脚本：
- 快速评估：[tools/evaluate_fast.sh](tools/evaluate_fast.sh)
- 标准评估：[tools/evaluate.sh](tools/evaluate.sh)

**常见环境变量与提示**
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（减少显存碎片）
- `CUDA_VISIBLE_DEVICES=0,1`（选择 GPU）
- 若遇到 `ModuleNotFoundError`：检查是否已激活 `recondreamerNew-rl` 并设置了 `PYTHONPATH`
- 若 CUDA 版本不匹配：请保持 Conda 中的 `cudatoolkit=11.8` 与驱动兼容

**参考文件与目录**
- 环境： [environment.yml](environment.yml)
- DiffusionDriveV2 环境： [DiffusionDriveV2/environment.yml](DiffusionDriveV2/environment.yml)
- 数据下载脚本： [DiffusionDriveV2/download](DiffusionDriveV2/download)
- 评估脚本： [tools/evaluate_fast.sh](tools/evaluate_fast.sh), [tools/evaluate.sh](tools/evaluate.sh)
- 训练（闭环RL示例）： [script/train_closed_loop.py](script/train_closed_loop.py)

以上为最小可用的安装与运行步骤。需要完整训练流程与更细致评估，请参考 DiffusionDriveV2 的文档与脚本目录。<div align="center">   

# ReconDreamer-RL: Enhancing Reinforcement Learning via Diffusion-based Scene Reconstruction


## [Project Page](https://github.com/GigaAI-research/ReconDreamer-RL) | [Paper](https://arxiv.org/html/2508.08170v1)
</div>

# Abstract 

Reinforcement learning for training end-to-end autonomous driving models in closed-loop simulations is gaining growing attention. However, most simulation environments differ significantly from real-world conditions, creating a substantial simulation-to-reality (sim2real) gap. To bridge this gap, some approaches utilize scene reconstruction techniques to create photorealistic environments as a simulator. While this improves realistic sensor simulation, these methods are inherently constrained by the distribution of the training data, making it difficult to render high-quality sensor data for novel trajectories or corner case scenarios. Therefore, we propose <strong>ReconDreamer-RL</strong>, a framework designed to integrate video diffusion priors into scene reconstruction to aid reinforcement learning, thereby enhancing end-to-end autonomous driving training. Specifically, in <strong>ReconDreamer-RL</strong>, we introduce <strong>ReconSimulator</strong>, which combines the video diffusion prior for appearance modeling and incorporates a kinematic model for physical modeling, thereby reconstructing driving scenarios from real-world data. This narrows the sim2real gap for closed-loop evaluation and reinforcement learning. To cover more corner-case scenarios, we introduce the <strong>Dynamic Adversary Agent (DAA)</strong>, which adjusts the trajectories of surrounding vehicles relative to the ego vehicle, autonomously generating corner-case traffic scenarios (e.g., cut-in). Finally, the <strong>Cousin Trajectory Generator (CTG)</strong> is proposed to address the issue of training data distribution, which is often biased toward simple straight-line movements. Experiments show that <strong>ReconDreamer-RL</strong> improves end-to-end autonomous driving training, outperforming imitation learning methods with a 5× reduction in the Collision Ratio.</p>


<img width="919" alt="abs" src="https://github.com/Nichaojun/Nichaojun.github.io/blob/main/images/pipeline6_01.png">


# News
- **[2025/11/1]** We provide the 3DGS reconstruction from the NuScenes dataset, along with a Gym-based environment for closed-loop simulation.


# Getting Started

**a. Create a Conda virtual environment and activate it.**

```shell
conda create -n recondreamer-rl python=3.9 -y
conda activate recondreamer-rl
```

**b. Install required packages.**

```shell
pip install -r requirements.txt
```

**c. Download necessary assets.**
To download the assets required for this project, execute the following command:

```shell
git clone https://huggingface.co/datasets/Ni1111/ReconDreamer-RL/tree/main/assets
mkdir  assets/third

git clone --branch v1.3.0 https://github.com/nerfstudio-project/gsplat.git assets/third/gsplat-1.3.0
git clone --branch v0.3.0 https://github.com/NVlabs/nvdiffrast.git assets/third/nvdiffrast-0.3.0
```
Final Directory Structure:
```text
project_root/
├── assets/
│   ├── third/
│   │   ├── gsplat-1.3.0/
│   │   └── nvdiffrast-0.3.0/
│   └── nus/
├── policy/
├── reconsimulator/
├── script/
└── requirements.txt
```

**d. Install third-party dependencies.**

```shell
pip install -e assets/third/gsplat-1.3.0
pip install -e assets/third/nvdiffrast-0.3.0
```

**e. Start the 3DGS environment as a server.**

```shell
conda activate recondreamer-rl
python script/eval_human_policy.py
```

**f. Launch the human policy.**
We provide an example of how to interact with the 3DGS environment using the human policy, which controls the ego vehicle based on the dataset annotations.
```shell
conda activate recondreamer-rl
python policy/human/deploy_policy.py
```


#  Citation
If you find Recondreamer-RL useful in your research or applications, please consider giving us a star and citing it by the following BibTeX entry:

```bibtex
@article{Recondreamer-RL, 
  title={Recondreamer-RL: Enhancing reinforcement learning via diffusion-based scene reconstruction},
  author={Ni, Chaojun and Zhao, Guosheng and Wang, Xiaofeng and Zhu, Zheng and Qin, Wenkang and Chen, Xinze and Jia, Guanghong and Huang, Guan and Mei, Wenjun},
  journal={arXiv preprint arXiv:2508.08170},
  year={2025}
}
````

#  Acknowledgments
Recondreamer-RL is greatly inspired by the following outstanding works:
* **[RAD](https://github.com/hustvl/RAD.git)**
* [VADv2](https://github.com/priest-yang/VADv2.git)
* [DriveStudio](https://github.com/ziyc/drivestudio.git)
* [DriveDreamer2](https://github.com/f1yfisher/DriveDreamer2.git)