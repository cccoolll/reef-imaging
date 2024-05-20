# REEF Imaging

REEF Imaging Farm Control Software

## Installation and Usage

First we need to clone the repository:
```

git clone git@github.com:aicell-lab/reef-imaging.git
cd reef-imaging
conda create -n reef-imaging python=3.11 -y
conda activate reef-imaging
# Install squid-control in editable mode
git clone git@github.com:aicell-lab/squid-control.git
pip install -e squid-control
pip install -r requirements.txt
pip install -e .
```

### Usage

To run the software, use the following command:
```
python -m reef_imaging.hypha_service
```
