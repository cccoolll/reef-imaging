# Mirror Services for Reef Imaging

This utility creates mirror services on hypha.aicell.io that forward requests to local services.

## Purpose

The mirror services script creates Hypha services on the cloud-based hypha.aicell.io server that transparently forward API calls to local services running on localhost. This allows remote clients to access all local services as if they were directly connected to the local server.

## How It Works

1. Connects to both the local Hypha server (localhost:9527) and the remote hypha.aicell.io server
2. Lists all available services on the local server
3. For each local service, creates a corresponding mirror service on hypha.aicell.io
4. Forwards all API calls from the mirror services to their local counterparts
5. Implements health checks to ensure mirrors reconnect if disconnected

## Prerequisites

Before running this script, ensure:

1. Local services are running on `localhost:9527`
2. Environment variables are set:
   - `REEF_LOCAL_TOKEN` - Token for authentication with the local Hypha server
   - `REEF_WORKSPACE_TOKEN` - Token for authentication with hypha.aicell.io

## Usage

Run the script with:

```bash
python reef_imaging/control/mirror_services.py
```

## Logs

The script logs its operations to `mirror_services.log` in the current directory.

## Service URLs

After starting, the script will output URLs for each mirrored service. These URLs can be shared with remote users or integrated with remote applications that need to access the local hardware.

Example:
```
Mirror service for microscope-control registered with ID: reef-imaging:microscope-control-mirror
Access the service at: https://hypha.aicell.io/reef-imaging/services/microscope-control-mirror
``` 