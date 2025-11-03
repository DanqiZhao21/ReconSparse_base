<div align="center">   

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