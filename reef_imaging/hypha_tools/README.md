# Hypha Tools

A collection of utilities for working with the [Artifact manager](https://docs.amun.ai/#/artifact-manager?id=artifact-manager) for reef imaging data processing and management.

## Core Components

- **Automated Uploaders**
  - `automated_treatment_uploader.py` - Uploads time-lapse experiment data from treatment folders
  - `automated_stitch_uploader.py` - Processes and uploads stitched image data as Zarr files

- **Storage Utilities**
  - `hypha_storage.py` - Provides a data store interface for the Hypha platform
  - `email_autorization.py` - Handles email-based authorization for Hypha services

- **Artifact Management**
  - `artifact_manager/` - Core library for interacting with Hypha's artifact management system
    - Connection handling
    - File uploading
    - Image stitching
    - Gallery management

## Data Files

- `treatment_upload_record.json` - Tracks uploaded treatment data
- `zarr_upload_record.json` - Tracks uploaded stitched image data
- `treatment_upload_progress.txt` - Simple log of treatment upload progress

## Usage

Most utilities are designed to run as standalone scripts and require environment variables for authentication:

```
REEF_WORKSPACE_TOKEN=your_token_here
```

Example:
```bash
python automated_treatment_uploader.py
```

## Dependencies

- hypha_rpc - Core library for Hypha RPC connections
- aiohttp - For asynchronous HTTP requests
- dotenv - For loading environment variables
- zarr - For working with Zarr format data
- pandas - For data processing
- opencv-python - For image processing

## Notes

These tools are designed for use with the reef-imaging project workflow, specifically for handling time-lapse microscopy data uploads to [Artifact manager](https://docs.amun.ai/#/artifact-manager?id=artifact-manager). 