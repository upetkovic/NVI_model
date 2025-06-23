#!/bin/bash

# Create a new conda environment named "labelling_env" with Python 3.10
conda create --name labelling_env python=3.10 -y

# Activate the newly created environment
source activate labelling_env

# Install OpenCV in the environment
conda install -c conda-forge opencv -y
pip install pyyaml
pip install -e .


echo "Environment 'labelling_env' created with Python 3.10 and OpenCV installed."