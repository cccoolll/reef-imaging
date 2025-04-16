"""
Examples of using the Reef Imaging Hypha tools.

This module provides examples of how to use the refactored classes for:
1. Creating galleries and datasets
2. Uploading zarr files
3. Uploading treatment data
4. Stitching images into zarr files
"""

import os
import asyncio
import pandas as pd

from .core import Config
from .gallery_manager import GalleryManager
from .uploader import ArtifactUploader
from .stitch_manager import StitchManager, ImageFileParser


async def create_gallery_example() -> None:
    """Example of creating a gallery and dataset"""
    gallery_manager = GalleryManager()
    
    try:
        # Create a gallery
        await gallery_manager.create_gallery(
            name="Image Map of U2OS FUCCI Drug Treatment",
            description="A collection for organizing imaging datasets acquired by microscopes",
            alias="reef-imaging/image-map-of-u2os-fucci-drug-treatment"
        )
        
        # Create a dataset in the gallery
        await gallery_manager.create_dataset(
            name="image-map-20250410-treatment",
            description="The Image Map of U2OS FUCCI Drug Treatment",
            alias="image-map-20250410-treatment",
            parent_id="reef-imaging/image-map-of-u2os-fucci-drug-treatment"
        )
    finally:
        await gallery_manager.connection.disconnect()


async def upload_zarr_example() -> None:
    """Example of uploading zarr files"""
    # Original zarr paths with .zarr extension
    ORIGINAL_ZARR_PATHS = [
        "/media/reef/harddisk/test_stitch_zarr/2025-04-10_13-50-7.zarr",
        "/media/reef/harddisk/test_stitch_zarr/2025-04-10_14-50-7.zarr"
    ]
    
    uploader = ArtifactUploader(
        artifact_alias="image-map-20250410-treatment",
        record_file="zarr_upload_record.json"
    )
    
    success = await uploader.upload_zarr_files(ORIGINAL_ZARR_PATHS)
    
    if success:
        # Commit the dataset if all files were uploaded successfully
        gallery_manager = GalleryManager()
        await gallery_manager.commit_dataset("image-map-20250410-treatment")
        await gallery_manager.connection.disconnect()


async def upload_treatment_example() -> None:
    """Example of uploading treatment data"""
    # List of source directories to upload
    SOURCE_DIRS = [
        "/media/reef/harddisk/20250410-fucci-time-lapse-scan_2025-04-10_13-50-7.762411",
        "/media/reef/harddisk/20250410-fucci-time-lapse-scan_2025-04-10_14-50-7.948398"
    ]
    
    uploader = ArtifactUploader(
        artifact_alias="20250410-treatment",
        record_file="treatment_upload_record.json"
    )
    
    success = await uploader.upload_treatment_data(SOURCE_DIRS)
    
    if success:
        # Commit the dataset if all files were uploaded successfully
        gallery_manager = GalleryManager()
        await gallery_manager.commit_dataset("20250410-treatment")
        await gallery_manager.connection.disconnect()


def stitch_images_example() -> None:
    """Example of stitching images"""
    # Paths and parameters
    data_folder = "/media/reef/harddisk/20250410-fucci-time-lapse-scan_2025-04-10_13-50-7.762411"
    image_folder = os.path.join(data_folder, "0")
    parameter_file = os.path.join(data_folder, "acquisition parameters.json")
    coordinates_file = os.path.join(image_folder, "coordinates.csv")
    output_folder = "/media/reef/harddisk/test_stitch_zarr"
    
    # Extract date and time from the folder name
    folder_name = os.path.basename(data_folder)
    date_time_str = folder_name.split('_')[1] + '_' + folder_name.split('_')[2].split('.')[0]
    zarr_filename = f"{date_time_str}.zarr"
    
    # Load coordinates
    coordinates = pd.read_csv(coordinates_file)
    
    # Predefined stage limits
    stage_limits = {
        "x_positive": 120,
        "x_negative": 0,
        "y_positive": 86,
        "y_negative": 0,
        "z_positive": 6
    }
    
    # Parse image filenames and get unique channels
    image_parser = ImageFileParser()
    image_info = image_parser.parse_image_filenames(image_folder)
    channels = list(set(info["channel_name"] for info in image_info))
    selected_channel = ['Fluorescence_561_nm_Ex', 'Fluorescence_488_nm_Ex', 'BF_LED_matrix_full']
    
    # Initialize stitch manager
    stitch_manager = StitchManager()
    
    # Setup from parameters
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


async def full_pipeline_example() -> None:
    """Example of the complete workflow: stitch, create gallery, upload"""
    # 1. Stitch images
    stitch_images_example()
    
    # 2. Create gallery and dataset
    await create_gallery_example()
    
    # 3. Upload stitched zarr files
    await upload_zarr_example()
    
    print("Full pipeline completed successfully!")


if __name__ == "__main__":
    import sys
    
    # Choose which example to run based on command line argument
    example = sys.argv[1] if len(sys.argv) > 1 else "full"
    
    if example == "gallery":
        asyncio.run(create_gallery_example())
    elif example == "zarr":
        asyncio.run(upload_zarr_example())
    elif example == "treatment":
        asyncio.run(upload_treatment_example())
    elif example == "stitch":
        stitch_images_example()
    elif example == "full":
        asyncio.run(full_pipeline_example())
    else:
        print(f"Unknown example: {example}")
        print("Available examples: gallery, zarr, treatment, stitch, full") 