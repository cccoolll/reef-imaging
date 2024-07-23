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

#### Start Hypha Server

Before start, make sure you installed docker and docker-compose.

 * Git clone this repo 
 * **IMPORTANT**: Change permission for HTTPS
   Run `chmod 600 traefik/acme/acme.json`
 * Create an `.env` file and fill in with keys defined in the template file `.env-template`. 
 * Change host name and other settings in the `docker-compose.yaml` file.
 * Create a docker network by running: `docker network create hypha-app-engine`
 * Start the application containers: `docker-compose up -d`
 * Start traefik: `cd traefik && docker-compose up -d`
 * Wait for a minutes and you should get your site running, to test it you can visit https://reef.aicell.io with your web browser.
  

#### Start Hypha Services

To run the software, use the following command:
```
python -m reef_imaging.hypha_service

```
(Setup python path for folder 'reef_imaging' and 'squid-control')