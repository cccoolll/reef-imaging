import os
import sys
import time
import asyncio
import json
import shutil
import re
from datetime import datetime
from typing import List, Optional, Tuple
from dotenv import load_dotenv
import cv2
import numpy as np
import pandas as pd
import zarr

# Import shared tools from artifact_manager
from artifact_manager.core import Config, HyphaConnection
from artifact_manager.uploader import ArtifactUploader
from artifact_manager.stitch_manager import StitchManager, ImageFileParser

# Constants
BASE_DIR = "/media/reef/harddisk"
EXPERIMENT_ID = "20250410-fucci-time-lapse-scan"
STITCHED_DIR = "/media/reef/harddisk/test_stitch_zarr"
STITCH_RECORD_FILE = "stitch_upload_progress.txt"
UPLOAD_RECORD_FILE = "zarr_upload_record.json"
CHECK_INTERVAL = 60  # Check for new folders every 60 seconds

# Load environment variables
load_dotenv()
ARTIFACT_ALIAS = "image-map-20250410-treatment-full"


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
    """Delete a zarr file."""
    zarr_path = os.path.join(STITCHED_DIR, zarr_file)
    try:
        print(f"Removing zarr file: {zarr_path}")
        if os.path.isdir(zarr_path):
            shutil.rmtree(zarr_path)
        else:
            os.remove(zarr_path)
        print(f"Successfully removed {zarr_path}")
    except Exception as e:
        print(f"Error removing {zarr_path}: {e}")


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


async def upload_zarr_file(zarr_file: str) -> bool:
    """Upload a zarr file to the artifact manager."""
    zarr_path = os.path.join(STITCHED_DIR, zarr_file)
    if not os.path.exists(zarr_path):
        print(f"Zarr file {zarr_path} does not exist")
        return False
    
    # Create uploader instance
    uploader = ArtifactUploader(
        artifact_alias=ARTIFACT_ALIAS,
        record_file=UPLOAD_RECORD_FILE
    )
    
    # Create a fresh connection
    connection = HyphaConnection()
    connection_task = None
    
    try:
        # Connect to Hypha
        connection_task = asyncio.create_task(connection.connect())
        await connection_task
        
        # Assign connection to uploader
        uploader.connection = connection
        
        # Put dataset in staging mode
        try:
            print(f"Putting dataset {ARTIFACT_ALIAS} in staging mode...")
            artifact_manager = connection.artifact_manager
            
            # Read the current manifest
            dataset = await artifact_manager.read(
                artifact_id=ARTIFACT_ALIAS,
                silent=True
            )
            
            # Put the dataset in staging mode
            await artifact_manager.edit(
                artifact_id=ARTIFACT_ALIAS,
                manifest=dataset,  # Preserve the same manifest
                version="stage"    # Put in staging mode
            )
            print("Dataset is now in staging mode")
            
        except Exception as e:
            print(f"Error putting dataset in staging mode: {e}")
            if "not found" in str(e).lower():
                print("Dataset not found. It may need to be created first.")
                return False
        
        # Add zarr path to the list of zarr files to upload
        zarr_paths = [zarr_path]
        
        # Upload the zarr file
        success = await uploader.upload_zarr_files(zarr_paths)
        
        # Commit the dataset
        if success:
            commit_success = False
            commit_attempts = 0
            
            while not commit_success and commit_attempts < 5:
                try:
                    # Refresh connection before commit
                    await connection.disconnect()
                    connection_task = asyncio.create_task(connection.connect())
                    await connection_task
                    
                    # Commit the dataset
                    await connection.artifact_manager.commit(ARTIFACT_ALIAS)
                    print("Dataset committed successfully.")
                    commit_success = True
                except Exception as e:
                    commit_attempts += 1
                    print(f"Error committing dataset (attempt {commit_attempts}/5): {str(e)}")
                    await asyncio.sleep(5)
                    
                    # Reset connection
                    if connection_task:
                        connection_task.cancel()
                    await connection.disconnect()
                    connection_task = asyncio.create_task(connection.connect())
                    await connection_task
            
            if not commit_success:
                print("WARNING: Failed to commit the dataset after multiple attempts.")
                return False
                
        return success
        
    except Exception as e:
        print(f"Error uploading zarr file: {e}")
        return False
    finally:
        # Clean up connection
        if connection_task and not connection_task.done():
            connection_task.cancel()
        await connection.disconnect()


async def process_folder(folder_name: str) -> bool:
    folder_path = os.path.join(BASE_DIR, folder_name)
    
    # Step 1: Check for .done file
    folder_datetime = extract_datetime_from_folder(folder_name)
    zarr_filename = f"{folder_datetime}.zarr" if folder_datetime else f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zarr"
    done_file_path = os.path.join(STITCHED_DIR, f"{zarr_filename}.done")
    if os.path.exists(done_file_path):
        print(f".done file found for {folder_name}, skipping stitching.")
    else:
        # Step 2: Stitch the folder
        print(f"\nStarting stitching process for folder: {folder_name}")
        stitch_success = stitch_folder(folder_path)
        if not stitch_success:
            print(f"Failed to stitch folder: {folder_name}")
            return False
    
    # Step 3: Get the stitched zarr file
    zarr_files = get_zarr_files()
    if not zarr_files:
        print(f"No zarr files were created from stitching folder: {folder_name}")
        return False
    
    # Find the matching zarr file for this folder
    folder_datetime = extract_datetime_from_folder(folder_name)
    matching_zarr = None
    
    if folder_datetime:
        matching_zarr = [f for f in zarr_files if folder_datetime in f]
        if not matching_zarr:
            # If no exact match, use the most recently created zarr file
            print(f"No matching zarr file found for datetime {folder_datetime}, using the first available.")
            matching_zarr = [zarr_files[0]]
    else:
        # If datetime can't be extracted, use the first zarr file
        print(f"Could not extract datetime from folder name, using the first available zarr file.")
        matching_zarr = [zarr_files[0]]
    
    if not matching_zarr:
        print(f"No zarr file available to upload for folder: {folder_name}")
        return False
    
    zarr_file = matching_zarr[0]
    print(f"Found matching zarr file: {zarr_file}")
    
    # Step 4: Upload the zarr file
    print(f"Starting upload process for zarr file: {zarr_file}")
    upload_success = await upload_zarr_file(zarr_file)
    if not upload_success:
        print(f"Failed to upload zarr file: {zarr_file}")
        return False
    
    # Step 5: Clean up the zarr file
    print(f"Cleaning up zarr file: {zarr_file}")
    cleanup_zarr_file(zarr_file)
    
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
    
    # Clean up any existing zarr files before starting
    print("Cleaning up any existing zarr files before starting")
    existing_zarr_files = get_zarr_files()
    for zarr_file in existing_zarr_files:
        cleanup_zarr_file(zarr_file)
    
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