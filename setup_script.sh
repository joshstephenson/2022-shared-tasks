#!/usr/bin/env bash

# Clone necessary repositories
git clone --recurse-submodules https://github.com/joshstephenson/2022-shared-tasks

# Start a Python virtual environment
python -m venv virtual
source virtual/bin/activate

# install required python packages
cd 2022-shared-tasks/segmentation/
pip install -r ./requirements.txt

echo "Run 'source ./bin/activate' and then proceed with training"

