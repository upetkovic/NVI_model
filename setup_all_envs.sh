#!/bin/bash

set -e  # exit on error

### -------- NVI ENVIRONMENT SETUP -------- ###
echo "Creating and activating nvi_env..."
conda create -n nvi_env2 python=3.9 -y
conda activate nvi_env2 || source activate nvi_env2
pip install gdown

# CUDA & PyTorch
conda install -c nvidia/label/cuda-11.7.0 cuda-toolkit=11.7 -y
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117

# Mamba and compilers
conda install -n base -c conda-forge mamba -y
mamba install gcc_linux-64=9.5.0 gxx_linux-64=9.5.0 -c conda-forge -y

# Segment & Track Anything setup
cd external/GroundingDINO
pip install -e .
cd ../segment_and_track_anything/sam
pip install -e .
cd ../../../

pip install opencv-python pycocotools matplotlib scikit-image
pip install gradio==3.39.0 gdown ffmpy
pip install timm==0.4.5
pip install wget
bash external/segment_and_track_anything/script/download_ckpt.sh
pip install transformers==4.26.1

# Correlation extension
cd external/Pytorch-Correlation-extension
python setup.py install
cd ../../

pip3 install scikit-learn
pip install pyyaml
pip install gdown
pip install pingouin
gdown --fuzzy "https://drive.google.com/file/d/1oDgIHJPRQc8v_djqx-xo18rk5seKRGIV/view?usp=sharing" -O checkpoints/gesture_loss002807corr0838acc849.pth
gdown --fuzzy "https://drive.google.com/file/d/1IGtQKSWGOMYfD6wJnPw6olSIadW1i9ki/view?usp=sharing" -O checkpoints/distance_loss001154corr0552.pth
pip install -e .  # install NVI

### -------- DINOV2 ENVIRONMENT SETUP -------- ###
echo "Creating and activating dinov2 env..."
conda create -n dinov22 python=3.9 -y
conda activate dinov22 || source activate dinov22
pip install gdown

conda install -c nvidia/label/cuda-11.7.0 cuda-toolkit=11.7 -y
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117

cd external/dinov2
pip install omegaconf torchmetrics==0.10.3 fvcore iopath
pip install xformers==0.0.18
pip install git+https://github.com/facebookincubator/submitit
pip install --extra-index-url https://pypi.nvidia.com cuml-cu11 mmcv-full==1.5.0 mmsegmentation==0.27.0
pip install -e .
cd ../../
pip install pyyaml
pip install -e .

### -------- HSEEMOTION ENVIRONMENT SETUP -------- ###
echo "Creating and activating hsemotion env..."
conda create -n hsemotion2 python=3.9 -y
conda activate hsemotion2 || source activate hsemotion2
pip install gdown

conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8 -y
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
pip install hsemotion facenet-pytorch==2.5.3 matplotlib==3.8.0 pillow==9.4.0 opencv-python==4.8.1.78
pip install timm==0.9.2
pip install pyyaml
pip install -e .

### -------- INSTALL FFMPEG (system-wide) -------- ###
echo "Installing FFmpeg..."
sudo apt update
sudo apt install ffmpeg -y

echo "All environments and dependencies set up!"
