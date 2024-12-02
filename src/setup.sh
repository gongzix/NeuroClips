#!/bin/bash
# Commands to setup a new conda environment and install all the necessary packages
# See the environment.yaml file for "conda env export > environment.yaml" after running this.

#set -e

conda create -n neuroclips python=3.11 -y
conda activate neuroclips

conda install numpy matplotlib tqdm scikit-image jupyterlab -y
conda install -c conda-forge accelerate -y

pip install --upgrade pip
pip install -r requirements.txt