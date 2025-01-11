# Diffusion Policy
参考于https://github.com/real-stanford/diffusion_policy?tab=readme-ov-file

## 🛝 克隆仓库
将 diffusion_policy 仓库克隆到本地机器：
git clone https://github.com/real-stanford/diffusion_policy.git
cd diffusion_policy

## 🧾安装环境
1. 安装 Mambaforge

wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh

运行安装脚本
bash Mambaforge-Linux-x86_64.sh

2. 安装 Mujoco 依赖项
sudo apt update
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

3. 使用 Mambaforge 创建并激活项目环境
mamba env create -f conda_environment.yaml
conda activate robodiff


## 数据下载与配置
1、下载训练数据
mkdir data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip
rm -f pusht.zip
cd ..
2、wget -O image_pusht_diffusion_policy_cnn.yaml https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/config.yaml


## 模型训练与评估
1、运行训练
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0

2、评估预训练模型
python eval.py --checkpoint data/outputs/epoch
