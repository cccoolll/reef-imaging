import os
import sys
import time
import asyncio
import json
import shutil
import re
import random
from datetime import datetime
from typing import List, Optional, Tuple
from dotenv import load_dotenv
import cv2
import numpy as np
import pandas as pd
import zarr
import tempfile

# Import shared tools from artifact_manager
from artifact_manager.core import Config, HyphaConnection
from artifact_manager.uploader import ArtifactUploader
from artifact_manager.stitch_manager import StitchManager, ImageFileParser

# Constants
BASE_DIR = "/media/reef/harddisk"
EXPERIMENT_ID = "20250429-scan-time-lapse"
STITCHED_DIR = "/media/reef/harddisk/test_stitch_zarr"
STITCH_RECORD_FILE = "stitch_upload_progress.txt"
UPLOAD_RECORD_FILE = "zarr_upload_record.json"
CHECK_INTERVAL = 60  # Check for new folders every 60 seconds
client_id = "reef-client-stitch-uploader"
# Load environment variables
load_dotenv()
ARTIFACT_ALIAS = "image-map-20250429-treatment"
# Timeout and retry settings
CONNECTION_TIMEOUT = 60  # Timeout for connection operations in seconds
OPERATION_TIMEOUT = 3600  # Timeout for Hypha operations in seconds (increased from 60)
MAX_RETRIES = 300  # Maximum retries for operations
# Optimized concurrency settings
BATCH_SIZE = 30  # Batch size for file uploads
CONCURRENCY_LIMIT = 25  # Concurrent upload limit


def get_timelapse_folders() -> List[str]:
    """Get all timelapse folders for the experiment from the base directory."""
    if not os.path.exists(BASE_DIR):
        print(f"Base directory {BASE_DIR} not found!")
        return []
    
    all_folders = os.listdir(BASE_DIR)
    # Filter folders that match the experiment ID pattern
    timelapse_folders = [
        folder for folder in all_folders 
        if folder.startswith(EXPERIMENT_ID) and os.path.isdir(os.path.join(BASE_DIR, folder))
    ]
    
    # Sort folders by timestamp
    timelapse_folders.sort()
    
    return timelapse_folders


def get_processed_folders() -> List[str]:
    """Get list of already processed folders from the record file."""
    processed = []
    if not os.path.exists(STITCH_RECORD_FILE):
        return processed
    
    with open(STITCH_RECORD_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line and " - " in line:
                parts = line.split(" - ", 1)
                if len(parts) == 2 and parts[1].strip():
                    folder = parts[1].split(" - ")[0] if " - " in parts[1] else parts[1]
                    processed.append(folder.strip())
    
    return processed


def save_processed_folder(folder_name: str, status: str = "completed"):
    """Save the processed folder to the record file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(STITCH_RECORD_FILE, "a") as f:
        f.write(f"{timestamp} - {folder_name} - {status}\n")


def get_zarr_files() -> List[str]:
    """Get list of zarr files in the stitched directory."""
    if not os.path.exists(STITCHED_DIR):
        return []
    
    return [f for f in os.listdir(STITCHED_DIR) if f.endswith('.zarr')]


def cleanup_zarr_file(zarr_file: str):
    """Delete a zarr file and its associated .done file."""
    zarr_path = os.path.join(STITCHED_DIR, zarr_file)
    done_file = os.path.join(STITCHED_DIR, f"{zarr_file}.done")
    
    try:
        print(f"Removing zarr file: {zarr_path}")
        if os.path.isdir(zarr_path):
            shutil.rmtree(zarr_path)
        else:
            os.remove(zarr_path)
        print(f"Successfully removed {zarr_path}")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")


def extract_datetime_from_folder(folder_name: str) -> Optional[str]:
    """Extract the datetime string from a folder name."""
    pattern = r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d+)"
    match = re.search(pattern, folder_name)
    if match:
        return match.group(1)
    return None


def stitch_folder(folder_path: str) -> bool:
    """Implement stitching functionality using the StitchManager class."""        
    try:
        # Extract folder name and create zarr filename
        folder_name = os.path.basename(folder_path)
        folder_datetime = extract_datetime_from_folder(folder_name)
        zarr_filename = f"{folder_datetime}.zarr" if folder_datetime else f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zarr"
        print(f"Will create zarr file: {zarr_filename} for folder: {folder_name}")
        
        # Paths to input data
        image_folder = os.path.join(folder_path, "0")
        parameter_file = os.path.join(folder_path, "acquisition parameters.json")
        coordinates_file = os.path.join(image_folder, "coordinates.csv")
        
        # Check inputs exist
        if not os.path.exists(image_folder):
            print(f"Image folder {image_folder} does not exist")
            return False
        if not os.path.exists(parameter_file):
            print(f"Parameter file {parameter_file} does not exist")
            return False
            
        # Load coordinates file or create if it doesn't exist
        if os.path.exists(coordinates_file):
            coordinates = pd.read_csv(coordinates_file)
        else:
            print(f"Coordinates file {coordinates_file} not found")
            return False
            
        # Stage limits (hardcoded for now)
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
        if not image_info:
            print("No image files found in the folder")
            return False
            
        channels = list(set(info["channel_name"] for info in image_info))
        print(f"Found {len(image_info)} images with {len(channels)} channels")
        selected_channel = ['Fluorescence_561_nm_Ex', 'Fluorescence_488_nm_Ex', 'BF_LED_matrix_full']
        # Filter selected channels to only include ones that are present
        selected_channel = [c for c in selected_channel if c in channels]
        print(f"Selected channels: {selected_channel}")
        
        if not selected_channel:
            print("No matching channels found in the data")
            return False
        
        # Initialize stitch manager
        stitch_manager = StitchManager()
        
        # Setup from parameters
        stitch_manager.setup_from_parameters(
            parameter_file=parameter_file,
            stage_limits=stage_limits,
            output_folder=STITCHED_DIR,
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
        
        print(f"Stitching successfully completed, zarr file created: {zarr_filename}")
        
        # Create a .done file to indicate successful stitching
        done_file_path = os.path.join(STITCHED_DIR, f"{zarr_filename}.done")
        with open(done_file_path, "w") as done_file:
            done_file.write("Stitching completed successfully.")
        print(f"Created done file: {done_file_path}")
        return True
    except Exception as e:
        print(f"Error during stitching: {e}")
        import traceback
        traceback.print_exc()
        return False


async def upload_zarr_channel(zarr_path: str, channel: str, uploader: ArtifactUploader) -> bool:
    """Upload a single channel from a zarr file to the artifact manager."""
    try:
        # Get the base name for the relative path
        zarr_file = os.path.basename(zarr_path)
        base_name = zarr_file.replace('.zarr', '')
            
        # The full path to the channel in the zarr file
        channel_path = os.path.join(zarr_path, channel)
        
        if not os.path.exists(channel_path):
            print(f"Channel directory {channel_path} not found in zarr file")
            return False
        
        # Set the relative path for upload - this will determine how it's stored in the artifact
        relative_path = f"{base_name}/{channel}.zip"
        print(f"Uploading channel {channel} as {relative_path}")
        
        # Use the zip_and_upload_folder method from the uploader
        success = await uploader.zip_and_upload_folder(
            folder_path=channel_path,
            relative_path=relative_path,
            delete_zip_after=True
        )
        
        if success:
            print(f"Successfully uploaded channel {channel}")
        else:
            print(f"Failed to upload channel {channel}")
            
        return success
            
    except Exception as e:
        print(f"Error uploading channel {channel}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def upload_zarr_file(zarr_file: str) -> bool:
    """Upload a zarr file to the artifact manager by zipping and uploading each channel separately."""
    zarr_path = os.path.join(STITCHED_DIR, zarr_file)
    if not os.path.exists(zarr_path):
        print(f"Zarr file {zarr_path} does not exist")
        return False
    
    # Create uploader instance with optimized concurrency
    uploader = ArtifactUploader(
        artifact_alias=ARTIFACT_ALIAS,
        record_file=UPLOAD_RECORD_FILE,
        client_id=client_id,
        concurrency_limit=CONCURRENCY_LIMIT
    )
    
    # Use the uploader's internal connection management
    connection = None  # We will get the connection object from the uploader after connecting

    try:
        # Connect to Hypha using the uploader's method
        connect_success = await uploader.connect_with_retry(client_id=client_id)
        if not connect_success:
            print("Failed to establish connection to Hypha via uploader")
            return False
        
        # Get the connection object from the uploader AFTER successful connection
        connection = uploader.connection 
        if not connection or not connection.artifact_manager:
             print("Failed to get a valid connection or artifact manager from the uploader.")
             # Attempt to disconnect cleanly if connection object exists
             if connection:
                 await connection.disconnect()
             return False

        # Put the dataset in staging mode with timeout
        await asyncio.wait_for(
            connection.artifact_manager.edit(
                artifact_id=ARTIFACT_ALIAS,
                stage=True,
                    ),
                    timeout=OPERATION_TIMEOUT
                )
        
        print(f"Dataset {ARTIFACT_ALIAS} put in staging mode")
        
        # Get the channels from the zarr file (top-level directories in zarr)
        channels = [d for d in os.listdir(zarr_path) if os.path.isdir(os.path.join(zarr_path, d))]
        print(f"Found channels in zarr file: {channels}")
        
        # Upload each channel separately as its own zip file
        all_success = True
        for channel in channels:
            print(f"Processing channel: {channel}")
            channel_success = await upload_zarr_channel(zarr_path, channel, uploader)
            if not channel_success:
                print(f"Failed to upload channel {channel}")
                all_success = False
                break
        
        if not all_success:
            print("Failed to upload all channels")
            if connection: await connection.disconnect()
            return False
        
        # Commit the dataset
        await connection.artifact_manager.commit(artifact_id=ARTIFACT_ALIAS)

            
        return True
        
    except Exception as e:
        print(f"Error during zarr file processing {zarr_file}: {e}")
        import traceback
        traceback.print_exc()
        # Ensure disconnect on unexpected errors
        if connection: await connection.disconnect()
        return False
    finally:
        # Ensure final disconnection 
        print(f"Ensuring final disconnection for {zarr_file} processing...")
        # Check if the uploader still holds a connection reference and disconnect it
        if uploader.connection:
            await uploader.connection.disconnect()
        # Also cancel any potentially lingering connection task
        if uploader.connection_task and not uploader.connection_task.done():
             uploader.connection_task.cancel()
             try:
                 await asyncio.wait_for(asyncio.shield(uploader.connection_task), timeout=1)
             except (asyncio.CancelledError, asyncio.TimeoutError):
                 pass
             uploader.connection_task = None
        print(f"Final disconnection completed for {zarr_file}.")


async def process_folder(folder_name: str) -> bool:
    folder_path = os.path.join(BASE_DIR, folder_name)
    
    # Step 1: Check for .done file to determine if stitching is needed
    folder_datetime = extract_datetime_from_folder(folder_name)
    zarr_filename = f"{folder_datetime}.zarr" if folder_datetime else f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zarr"
    zarr_path = os.path.join(STITCHED_DIR, zarr_filename)
    done_file_path = os.path.join(STITCHED_DIR, f"{zarr_filename}.done")
    
    # If .done exists but zarr file doesn't, we need to restitch
    if os.path.exists(done_file_path) and not os.path.exists(zarr_path):
        print(f".done file found but zarr file missing. Restitching {folder_name}.")
        os.remove(done_file_path)  # Remove stale .done file
    
    # Stitch if needed
    if not os.path.exists(done_file_path) or not os.path.exists(zarr_path):
        print(f"\nStarting stitching process for folder: {folder_name}")
        stitch_success = stitch_folder(folder_path)
        if not stitch_success:
            print(f"Failed to stitch folder: {folder_name}")
            return False
    else:
        print(f".done file and zarr file found for {folder_name}, skipping stitching.")
    
    # Step 2: Verify zarr file exists
    if not os.path.exists(zarr_path):
        print(f"Zarr file {zarr_filename} not found after stitching. Something went wrong.")
        return False
    
    # Step 3: Upload the zarr file
    print(f"Starting upload process for zarr file: {zarr_filename}")
    upload_success = await upload_zarr_file(zarr_filename)
    if not upload_success:
        print(f"Failed to upload zarr file: {zarr_filename}")
        return False
    
    # Step 4: Record as processed
    save_processed_folder(folder_name)
    print(f"Successfully processed folder: {folder_name}")
    
    # Step 5: Clean up the zarr file AFTER recording as processed
    print(f"Cleaning up zarr file: {zarr_filename}")
    cleanup_zarr_file(zarr_filename)
    
    return True


async def main():
    # Ask user which folder to start with
    all_folders = get_timelapse_folders()
    
    if not all_folders:
        print(f"No folders matching {EXPERIMENT_ID} found in {BASE_DIR}")
        return
    
    processed_folders = get_processed_folders()
    
    print("Available folders:")
    for i, folder in enumerate(all_folders):
        status = " (processed)" if folder in processed_folders else ""
        print(f"{i+1}: {folder}{status}")
    
    # Default to the first unprocessed folder, or second-to-last if all processed
    unprocessed = [f for f in all_folders if f not in processed_folders]
    
    if not unprocessed and len(all_folders) >= 2:
        default_folder = all_folders[-2]  # Second to last if all processed
    elif unprocessed:
        default_folder = unprocessed[0]
    else:
        print("No suitable folders found to process")
        return
    
    # Ask for start folder
    user_input = input(f"Which folder do you want to start with? (default: {default_folder}): ")
    
    # If user provided input, use it; otherwise use default
    if user_input.strip():
        # Check if user input is an index or folder name
        if user_input.isdigit() and 1 <= int(user_input) <= len(all_folders):
            start_folder = all_folders[int(user_input) - 1]
        else:
            # Try to find the folder by name
            if user_input in all_folders:
                start_folder = user_input
            else:
                print(f"Folder {user_input} not found. Using default: {default_folder}")
                start_folder = default_folder
    else:
        start_folder = default_folder
    
    # Ask for end folder (optional)
    if len(all_folders) > 1:
        default_end_folder = all_folders[-2]  # Default to second-to-last folder
        user_input = input(f"Which folder do you want to end with? (default: {default_end_folder}, press Enter to process until the latest folder): ")
        
        if user_input.strip():
            # Check if user input is an index or folder name
            if user_input.isdigit() and 1 <= int(user_input) <= len(all_folders):
                end_folder = all_folders[int(user_input) - 1]
            else:
                # Try to find the folder by name
                if user_input in all_folders:
                    end_folder = user_input
                else:
                    print(f"Folder {user_input} not found. Using default: {default_end_folder}")
                    end_folder = default_end_folder
        else:
            end_folder = default_end_folder
    else:
        end_folder = start_folder
    
    start_idx = all_folders.index(start_folder)
    end_idx = all_folders.index(end_folder)
    
    # Validate indexes
    if start_idx > end_idx:
        print(f"Start folder ({start_folder}) comes after end folder ({end_folder}). Swapping them.")
        start_idx, end_idx = end_idx, start_idx
        start_folder, end_folder = end_folder, start_folder
    
    print(f"Processing folders from {start_folder} to {end_folder}")
    print(f"Starting with folder index {start_idx+1} and ending with folder index {end_idx+1}")
    
    # Ensure stitched directory exists
    os.makedirs(STITCHED_DIR, exist_ok=True)
    
    
    # Process the specified range of folders
    current_idx = start_idx
    
    while current_idx <= min(end_idx, len(all_folders) - 2):  # Don't process the last folder which might be incomplete
        folder_to_process = all_folders[current_idx]
        
        # Skip if already processed
        if folder_to_process in processed_folders:
            print(f"Folder {folder_to_process} already processed, skipping.")
            current_idx += 1
            continue
        
        print(f"\nProcessing folder: {folder_to_process} (index {current_idx+1})")
        
        # Process the folder
        success = await process_folder(folder_to_process)
        
        if success:
            save_processed_folder(folder_to_process)
            print(f"Successfully processed folder: {folder_to_process}")
            current_idx += 1
        else:
            # If failed, retry after a delay
            print(f"Failed to process {folder_to_process}. Retrying in {CHECK_INTERVAL} seconds...")
            await asyncio.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main()) 