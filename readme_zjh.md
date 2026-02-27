# 1 train diff pass
bash tools/train

调试,单独启动 actor 和 leaner
export PYTHONPATH=.
python script/train_actor_learner_v2.py

# 2 eval

bash tools/smalltool/visualize/generate_video.sh --scene=7 --duration-s 18.0 --step-frames 5

python /root/clone/ReconDreamer-RL/tools/smalltool/visualize/generate_video.py --scene 2 --ckpt /root/clone/ReconDreamer-RL/DiffusionDriveV2/ckpt/diffusiondrivev2_rl.ckpt --out /root/clone/ReconDreamer-RL/outputs/visualize/scene001_diffusiondrivev2_rl_20260226-051033.mp4


==== generate_video (single-scene) ====
scene=7 ckpt=/root/clone/ReconDreamer-RL/outputs/actor_learner/weights/latest.ckpt
device=cuda:0 cuda=0 debug=False
out=/root/clone/ReconDreamer-RL/outputs/visualize/scene007_latest_20260226-020600.mp4

# 3 eval  sparsedrive 
将sparsedrive 接入到framework中，加载sparsedrive的ckpt 作为policy进行评估

注意以下 config内的pkl， 需要和split的val的scene id 一样,如下命令是正确的，分支
python tools/test.py projects/configs/sparsedrive_small_stage2_reconic_zjh_001.py ckpt/sparsedrive_stage2.pth --eval bb
ox
可视化
python tools/visualization/visualize.py projects/configs/sparsedrive_small_stage2_reconic_zjh_001.py --result-path work_dirs/sparsedrive_small_stage2_reconic_zjh_001/results.pkl --out-dir work_dirs/vis/

```

env
torch  2.1.0+cu118
Python 3.10.19
mmcv-full mmcv_full-1.7.0+torch2.1.0cu118-cp310-cp310-win_amd64.whl
pip install mmcv_full=='1.7.0+torch2.1
.0cu118' -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

难点
flash_attn 需要手动下载对应版本 后 安装，目前下载在ckpt下，
https://github.com/Dao-AILab/flash-attention/releases/tag/v2.3.2

但是安装后编译torch和flash-attn的环境不一致, 注意一定要下载 False 的版本 abiFALSE不会依赖编译环境
python -c "import flash_attn_2_cuda; print('导入成功')"

还有报错libc10.so ,添加依赖，其中路径通过以下命令查看
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/root/miniconda3/envs/recondreamerNew-rl/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
 python -c "import flash_attn_2_cuda; print('导入成功')"
导入成功

python -c "
import torch
import os
torch_lib_path = os.path.dirname(torch.__file__) + '/lib'
print('PyTorch 库文件路径：', torch_lib_path)
print('libc10.so 是否存在：', os.path.exists(os.path.join(torch_lib_path, 'libc10.so')))
"



# 如果替换gcc 按如下流程
查看pytroch编译环境gcc
python -c "import torch; print('PyTorch 编译信息：\n', torch.__config__.show())"
PyTorch 编译信息：
 PyTorch built with:
  - GCC 9.3
  - C++ Version: 201703
  - Intel(R) oneAPI Math Kernel Library Version 2022.2-Product Build 20220804 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v3.1.1 (Git Hash 64f6bcbcbab628e96f33a62c3e975f8535a7bde4)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX512
  - CUDA Runtime 11.8
  - NVCC architecture flags: -gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_90,code=sm_90
  - CuDNN 8.7
  - Magma 2.6.1

重新安装g++
apt remove -y gcc g++
apt update && apt install -y gcc-9 g++-9
设置默认gcc 的软链接
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 100

# 2. 创建 c++ 到 g++-9 的软链接
ln -s /usr/bin/g++-9 /usr/bin/c++
ln -s /usr/bin/gcc-9 /usr/bin/cc


pip list | grep num
numba                      0.60.0
numpy                      1.26.4
替换numpy
pip uninstall numpy
pip install numpy==1.23.5
```


# 4 RL train sparsedrive
