

# ReconDiff简版使用说明
- **目标:** 统一使用本仓库，快速完成环境配置与数据下载；以 [environment.yml](environment.yml) 为准搭建统一环境（CUDA 11.8 + Python 3.10）。
- **内容:** 克隆与环境、数据与模型下载、快速运行（缓存与评估）、常见环境变量与提示。

**前置要求**
- NVIDIA GPU（建议 CUDA 11.8），已安装驱动
- Conda（Miniconda/Anaconda）与 Git

**获取代码**
- 克隆仓库并进入目录：

```bash
git clone https://github.com/DanqiZhao21/ReconDiff.git
cd ReconDiff
```

**配置环境（按 environment.yml）**
- 用根目录的环境文件创建并激活：

```bash
conda env create -f environment.yml
conda activate recondiff
```
<!-- 
- 推荐设置 `PYTHONPATH` 以确保包可见：

```bash
export PYTHONPATH=$(pwd):$(pwd)/DiffusionDriveV2:$(pwd)/DiffusionDriveV2/navsim:$PYTHONPATH
```

- 可选：以可编辑方式注册 DiffusionDriveV2（便于开发调试）：

```bash
pip install -e DiffusionDriveV2
``` -->

# 数据与模型准备

本文档说明在 **Denso 服务器环境** 下，本项目所依赖的数据集、预训练模型以及仿真资产的配置方式。  
由于相关资源已统一下载并集中存放在 `OpenDataset` 盘中，用户 **无需重复下载数据或模型**，只需按照说明确认环境变量与软连接配置即可。

---

## 一、navsim 数据集

本项目使用的 navsim 数据集，其安装与目录组织方式参考 DiffusionDriveV2 / navsim 官方说明：

- 官方文档：https://github.com/autonomousvision/navsim/blob/main/docs/install.md

在 Denso 服务器中：

- navsim **完整数据集与地图文件** 已全部下载完成  
- 数据统一存放在 `OpenDataset` 盘  
- 相关路径已通过 `bash` 环境变量提前配置完成  

请确认以下环境变量已存在（**无需修改**）：

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/OpenDataset/navsim/dataset/maps"
export NAVSIM_EXP_ROOT="/OpenDataset/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/OpenDataset/navsim/navsim"
export OPENSCENE_DATA_ROOT="/OpenDataset/navsim/dataset"
```
在上述环境变量正确设置的情况下，即可直接使用 navsim 数据集进行训练与评测。

## 二、DiffusionDriveV2 预训练模
DiffusionDriveV2 依赖 GTRS 轨迹生成模型。Denso服务器上已经下载放入OpenData盘 ,请自行进行软连接，结构应如下：

```bash
(recondreamerNew-rl)root@di-20251204200609-rw5zr:~/clone/ReconDreamer-RL# ls -l /root/clone/ReconDreamer-RL/DiffusionDriveV2/gtrs_traj
lrwxrwxrwx 1 root root 44 Dec 26 15:38 /root/clone/ReconDreamer-RL/DiffusionDriveV2/gtrs_traj -> /OpenDataset/DiffusionDriveV2_data/gtrs_traj
```

DiffusionDriveV2的预训练模型，已经下载放入OpenData盘 ,请自行进行软连接，结构应如下：
```bash
(recondreamerNew-rl)root@di-20251204200609-rw5zr:~/clone/ReconDreamer-RL# ls -l /root/clone/ReconDreamer-RL/DiffusionDriveV2/ckpt
lrwxrwxrwx 1 root root 39 Dec 27 01:30 /root/clone/ReconDreamer-RL/DiffusionDriveV2/ckpt -> /OpenDataset/DiffusionDriveV2_data/ckpt
```
## 三、Recon Simulator 的 3D 资产

Recon Simulator 所需的 3D 场景与资产文件同样已在 Denso 服务器中统一准备完成，并已配置好软连接。
在项目根目录下，assets 目录指向共享数据盘中的真实资产路径：
```bash
(recondreamerNew-rl) root@server:~/clone/ReconDreamer-RL# ls -l assets
lrwxrwxrwx 1 root root 35 Dec 26 15:34 assets \
-> /OpenDataset/ReconDreamer-RL/assets
```

# 快速运行
## 一、DiffusionDriveV2 快速测评

为加快验证流程，项目提供了 **快速测评（fast）** 版本的评测脚本及其对应的 cache 生成脚本。  
快速测评主要用于功能验证和调试，不作为最终性能报告。

### 1. 生成快速测评所需的 Cache

在运行快速测评前，需要先生成对应的 cache 文件：

```bash
bash /root/clone/ReconDreamer-RL/tools/cache_fast.sh
```
该脚本仅针对 DiffusionDriveV2 单模型评测，用于生成快速测评所需的 cache 数据。
### 2. 运行 DiffusionDriveV2 快速测评
完成 cache 生成后，可直接运行快速测评脚本：
```bash
bash /root/clone/ReconDreamer-RL/tools/evaluate_fast.sh
```
## 二、DiffusionDriveV2 标准测评

标准测评用于获取完整、可复现的评测结果，运行时间较长，但评测流程与指标设置更加全面。

### 1. 生成标准测评所需的 Cache

在运行标准测评前，需要先生成完整评测流程所需的 cache 文件：
```bash
bash /root/clone/ReconDreamer-RL/tools/cache.sh
```
### 2. 运行 DiffusionDriveV2 标准测评
完成 cache 生成后，执行标准评测脚本：
```bash
bash /root/clone/ReconDreamer-RL/tools/evaluate.sh
```
该评测流程对应 DiffusionDriveV2 的标准评测设置，推荐用于实验结果统计与对比分析.

## 三、闭环训练ReconDiff（DiffusionDriveV2 + ReconDreamer 3DGS）
本项目支持将 DiffusionDriveV2 与 ReconDreamer 的 3D Gaussian Splatting（3DGS） 结合，进行闭环训练。
运行以下脚本即可启动闭环训练流程：
```bash
bash /root/clone/ReconDreamer-RL/tools/trainclosedloop.sh
```
该脚本集成了：

- DiffusionDriveV2 的轨迹生成能力

- ReconDreamer 3DGS 的环境重建与感知反馈

- 闭环交互式训练流程

适用于端到端的闭环强化学习与仿真训练实验。

| 脚本名称 | 功能说明 |
| --- | --- |
| cache_fast.sh | DiffusionDriveV2 快速测评所需 cache 生成 |
| evaluate_fast.sh | DiffusionDriveV2 快速测评 |
| cache.sh | DiffusionDriveV2 标准测评所需 cache 生成 |
| evaluate.sh | DiffusionDriveV2 标准测评 |
| trainclosedloop.sh | DiffusionDriveV2 + ReconDreamer 3DGS 闭环训练 |


# 参考文件与目录
- 环境： [environment.yml](environment.yml)
- DiffusionDriveV2 ： [DiffusionDriveV2](https://github.com/hustvl/DiffusionDriveV2)
- 3DGS World Model : [ReconDreamer-RL](https://github.com/GigaAI-research/ReconDreamer-RL)
- 数据下载脚本： [Navsim](https://github.com/autonomousvision/navsim/blob/main/docs/install.md)

   
