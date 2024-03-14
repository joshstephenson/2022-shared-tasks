#!/usr/bin/env bash

# Clone necessary repositories
git clone --recursive-submodules https://github.com/joshstephenson/2022-shared-tasks
#git clone https://github.com/joshstephenson/fairseq-entmax

# Start a Python virtual environment
python -m venv virtual
source virtual/bin/activate

# install required python packages
cd 2022-shared-tasks/segmentation/
pip install -r ./requirements.txt


