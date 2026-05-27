git add .
git status
git restore --staged gtrs_traj
git commit -m "grpo infra"
git remote set-url origin git@github.com:DanqiZhao21/ReconDiff.git#使用ssh
git push origin main


#关于工作树提交
cd /root/clone/ReconDreamer-RL
git merge framework-architecture-cleanup