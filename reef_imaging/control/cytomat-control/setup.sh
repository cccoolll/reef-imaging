#!/bin/bash

# Create a conda environment named 'cytomat-env' with Python 3.8
conda create --name cytomat-env python=3.10 -y

# Activate the conda environment
source activate cytomat-env

# Install the required packages using pip
pip install -e ../../..

# Run the setup_cytomat script
python -m cytomat.scripts.setup_cytomat