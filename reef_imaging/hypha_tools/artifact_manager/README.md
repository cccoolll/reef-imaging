# Reef Imaging Hypha Tools

This package provides a set of classes and utilities for working with the Hypha Artifact Manager, specifically designed for the Reef Imaging project.

## Core Features

- **Gallery Management**: Create and manage galleries and datasets
- **File Uploading**: Upload files to the Hypha Artifact Manager with robust error handling and retries
- **Treatment Data Handling**: Upload and manage scientific treatment data
- **Channel-Based Upload**: Upload zarr channels as individual zip files for optimized data access

## Workflow Overview

The package includes two main upload workflows:

1. **Treatment Upload**: Uploads raw experiment data directly to Hypha
2. **Stitch Upload**: Stitches microscopy images into zarr files before uploading

Both workflows use optimized connection handling with retry logic, concurrent batch uploads, and progress tracking.

### Zarr Channel Upload Workflow

The latest version implements a channel-based upload approach for zarr files:
1. The stitcher creates a zarr file with multiple channels (e.g., BF_LED_matrix_full, Fluorescence_488_nm_Ex, Fluorescence_561_nm_Ex)
2. Each channel is uploaded as a separate zip file with the pattern `{timestamp}/{channel}.zip`
3. This approach allows for more efficient data retrieval and better fault tolerance during uploads

![Reef Imaging Upload Process Architecture](docs/upload_process_diagram.png)

To generate the workflow diagram from the included DOT file:

```bash
# Install Graphviz if needed
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Generate PNG image from DOT file
dot -Tpng upload_process.dot -o upload_process_diagram.png
```

The diagram visualizes how both uploaders share common connection management while handling different types of scientific data processing.

## Installation

The package is designed to be used within the Reef Imaging codebase. No additional installation is required.

## Usage

Here are some examples of how to use the package:

### Create a Gallery and Dataset

```python
import asyncio
from reef_imaging.hypha_tools.artifact_manager import GalleryManager

async def create_gallery():
    gallery_manager = GalleryManager()
    
    try:
        # Create a gallery
        await gallery_manager.create_gallery(
            name="My Gallery",
            description="A collection of datasets",
            alias="my-gallery"
        )
        
        # Create a dataset
        await gallery_manager.create_dataset(
            name="my-dataset",
            description="A dataset in the gallery",
            alias="my-dataset",
            parent_id="my-gallery"
        )
    finally:
        await gallery_manager.connection.disconnect()

# Run the async function
asyncio.run(create_gallery())
```

### Upload Zarr Channels

```python
import asyncio
import os
from reef_imaging.hypha_tools.artifact_manager import ArtifactUploader

async def upload_zarr_channels():
    # Path to zarr file
    zarr_path = "/path/to/2025-04-29_15-38-36.zarr"
    
    uploader = ArtifactUploader(
        artifact_alias="agent-lens/image-map-20250429-treatment",
        record_file="upload_record.json"
    )
    
    # Connect to Hypha
    await uploader.connect_with_retry()
    
    try:
        # Get all channels from the zarr file
        channels = [d for d in os.listdir(zarr_path) 
                   if os.path.isdir(os.path.join(zarr_path, d))]
        
        # Upload each channel as a separate zip file
        for channel in channels:
            base_name = os.path.basename(zarr_path).replace('.zarr', '')
            channel_path = os.path.join(zarr_path, channel)
            relative_path = f"{base_name}/{channel}.zip"
            
            success = await uploader.zip_and_upload_folder(
                folder_path=channel_path,
                relative_path=relative_path,
                delete_zip_after=True
            )
            
            if success:
                print(f"Channel {channel} uploaded successfully!")
            else:
                print(f"Failed to upload channel {channel}")
    finally:
        if uploader.connection:
            await uploader.connection.disconnect()

# Run the async function
asyncio.run(upload_zarr_channels())
```

### Upload Treatment Data

```python
import asyncio
from reef_imaging.hypha_tools.artifact_manager import ArtifactUploader

async def upload_treatment_data():
    # List of source directories to upload
    source_dirs = [
        "/path/to/experiment1",
        "/path/to/experiment2"
    ]
    
    uploader = ArtifactUploader(
        artifact_alias="treatment-dataset",
        record_file="treatment_upload_record.json"
    )
    
    success = await uploader.upload_treatment_data(source_dirs)
    
    if success:
        print("Treatment data uploaded successfully!")
    else:
        print("Some treatment data failed to upload")

# Run the async function
asyncio.run(upload_treatment_data())
```

### Stitch Images

```python
import os
import pandas as pd
from reef_imaging.hypha_tools.artifact_manager import StitchManager, ImageFileParser

def stitch_images():
    # Define paths and parameters
    data_folder = "/path/to/data_folder"
    image_folder = os.path.join(data_folder, "0")
    parameter_file = os.path.join(data_folder, "acquisition parameters.json")
    coordinates_file = os.path.join(image_folder, "coordinates.csv")
    output_folder = "/path/to/output"
    zarr_filename = "stitched.zarr"
    
    # Load coordinates
    coordinates = pd.read_csv(coordinates_file)
    
    # Stage limits
    stage_limits = {
        "x_positive": 120,
        "x_negative": 0,
        "y_positive": 86,
        "y_negative": 0,
        "z_positive": 6
    }
    
    # Parse image filenames
    image_parser = ImageFileParser()
    image_info = image_parser.parse_image_filenames(image_folder)
    
    # Get channels
    channels = list(set(info["channel_name"] for info in image_info))
    selected_channel = channels  # Or select specific channels
    
    # Initialize stitch manager
    stitch_manager = StitchManager()
    
    # Setup
    stitch_manager.setup_from_parameters(
        parameter_file=parameter_file,
        stage_limits=stage_limits,
        output_folder=output_folder,
        zarr_filename=zarr_filename,
        channels=selected_channel,
        pyramid_levels=6
    )
    
    # Process images
    stitch_manager.process_images(
        image_info=image_info,
        coordinates=coordinates,
        selected_channel=selected_channel,
        pyramid_levels=6
    )
    
    print(f"Stitching complete. Output saved to {os.path.join(output_folder, zarr_filename)}")

# Run the function
stitch_images()
```

## Examples

You can run these examples from the command line:

```bash

# Run specific examples
python -m reef_imaging.hypha_tools.artifact_manager.examples gallery
python -m reef_imaging.hypha_tools.artifact_manager.examples zarr
python -m reef_imaging.hypha_tools.artifact_manager.examples treatment
python -m reef_imaging.hypha_tools.artifact_manager.examples stitch
```

## Configuration

The package uses environment variables for configuration. The following variables are required:

- `REEF_WORKSPACE_TOKEN`: Authentication token for the Hypha server

You can set these variables in a `.env` file or in your environment.

## Error Handling

The upload functionality includes robust error handling with:

- Automatic retries with exponential backoff
- Connection timeouts
- Progress tracking
- Resume capability for interrupted uploads

## Package Structure

The package is organized into the following modules:

- `core.py`: Core utilities like configuration and connection management
- `gallery_manager.py`: Gallery and dataset management
- `uploader.py`: File uploading with robust error handling
- `stitch_manager.py`: Image stitching and processing utilities

## Available Classes

The package exports the following classes:

- `Config`: Configuration management using environment variables
- `HyphaConnection`: Connection to the Hypha server
- `UploadRecord`: Track and resume file uploads
- `GalleryManager`: Create and manage galleries and datasets
- `ArtifactUploader`: Upload files with robust error handling
- `StitchManager`: Stitch microscopy images into zarr files
- `ImageFileParser`: Parse image filenames to extract metadata
- `ImagingParameters`: Process microscopy imaging parameters
- `ImageProcessor`: Process individual images
- `StitchCanvas`: Create a canvas for stitching images
- `ZarrWriter`: Write data to zarr files

## Contributing

To contribute to this package:

1. Make sure your changes follow the object-oriented design principles
2. Add appropriate documentation and type hints
3. Test your changes with the example scripts 