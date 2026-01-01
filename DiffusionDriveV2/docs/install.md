# Installation for DiffusionDriveV2

After successfully installing the NAVSIM environment, you should further proceed to install the following packages for DiffusionDriveV2:

```bash
conda activate navsim
pip install diffusers einops 
```

To enable faster download, [Jinkun](https://github.com/Jzzzi) provide a improved script [super_download.sh](../download/super_download.sh) by using tmux to parallelize the download process. Thanks for his contribution!

```bash
cd /path/to/DiffusionDriveV2/download
bash super_download.sh
```

To utilize GTRS trajectories as data augmentation during mode selector training, please download the simulated ground-truths for the GTRS vocabulary.

```bash 
cd gtrs_traj
wget https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/navtrain_16384.pkl
```

