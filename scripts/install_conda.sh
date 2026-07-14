#!/bin/bash

# Install wget (assuming you're on a Debian-based system)
apt-get update
apt-get install wget -y

# Install Conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

# Initialize Conda
~/miniconda3/bin/conda init bash