"""tools/trytry.py

这是一个“命令备忘录”，原本文件里是 bash 片段，直接 `python tools/trytry.py` 会报语法错。

现在改成：运行本文件会打印一份可直接复制执行的 bash 命令（默认按仓库路径自动推导）。

可选环境变量：
- `NUPLAN_DEVKIT_ROOT`：nuplan-devkit 路径（如果你的 navsim 依赖它）。
- `NAVSIM_EXP_ROOT`：navsim cache/experiment 根目录。
"""
#你之后可以直接这样跑全流程：
cd /root/clone/ReconDreamer-RL
bash script/run_train_eval_pipeline.sh

#你之后可以直接这样跑全流程：
cd /root/clone/ReconDreamer-RL
bash script/run_train_eval_pipeline.sh --skip-reinforcepp

#如果你想先小规模试跑，比如只跑前 4 个场景：
cd /root/clone/ReconDreamer-RL
bash script/run_train_eval_pipeline.sh --max-scenes 4
