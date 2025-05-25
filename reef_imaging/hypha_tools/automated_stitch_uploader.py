import os
import sys
import time
import asyncio
import json
import shutil
import re
import random
import multiprocessing
from datetime import datetime
from typing import List, Optional, Tuple, Dict
from dotenv import load_dotenv
import cv2
import numpy as np
import pandas as pd
import zarr
import tempfile

# Import shared tools from artifact_manager
from artifact_manager.core import Config, HyphaConnection
from artifact_manager.uploader import ArtifactUploader
from artifact_manager.stitch_manager import StitchManager, ImageFileParser, add_chunk_metadata
from artifact_manager.gallery_manager import GalleryManager

# Constants
BASE_DIR = "/media/reef/harddisk"
EXPERIMENT_ID = "20250429-scan-time-lapse"
# Replace hardcoded STITCHED_DIR with temp directory
STITCHED_DIR = tempfile.mkdtemp(prefix="stitch_zarr_")
STITCH_RECORD_FILE = "stitch_upload_progress.txt"
CHECK_INTERVAL = 15  # Check for new files every 15 seconds
STABILITY_WINDOW = 5  # Consider folder stable after 5 seconds of no changes
client_id = "reef-client-stitch-uploader"
# Load environment variables
load_dotenv()
ARTIFACT_ALIAS = "agent-lens/image-map-20250429-treatment-zip"
# Timeout and retry settings
CONNECTION_TIMEOUT = 60  # Timeout for connection operations in seconds
OPERATION_TIMEOUT = 3600  # Timeout for Hypha operations in seconds (increased from 60)
MAX_RETRIES = 300  # Maximum retries for operations
# Optimized concurrency settings
BATCH_SIZE = 30  # Batch size for file uploads
CONCURRENCY_LIMIT = 25  # Concurrent upload limit

# Gallery and dataset settings
GALLERY_ALIAS = f"agent-lens/{EXPERIMENT_ID}-gallery"
GALLERY_NAME = f"Image Map of {EXPERIMENT_ID}"
GALLERY_DESCRIPTION = f"A collection for organizing imaging datasets from {EXPERIMENT_ID}"

# Global connection objects
gallery_manager = None
uploader = None

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
    parts = folder_name.split('_')
    if len(parts) >= 3:
        # Format: 20250429'-scan-time-lapse_2025-04-29_17-53-2.861421
        date_part = parts[1]  # Extract date part: 2025-04-29
        time_part = parts[2].split('.')[0]  # Extract time part without microseconds: 17-53-2
        time_components = time_part.split('-')
        if len(time_components) == 3:
            # Pad single-digit seconds with leading zero
            time_components[2] = time_components[2].zfill(2)
            time_part = '-'.join(time_components)
        return date_part + '_' + time_part
    return None


async def is_folder_stable(folder_path: str) -> bool:
    """Check if a folder is stable (no changes for STABILITY_WINDOW seconds)."""
    try:
        # Get initial modification time
        initial_mtime = os.path.getmtime(folder_path)
        
        # Wait for stability window
        await asyncio.sleep(STABILITY_WINDOW)
        
        # Check if modification time has changed
        current_mtime = os.path.getmtime(folder_path)
        
        # Also check if coordinates.csv exists
        coordinates_file = os.path.join(folder_path, "0", "coordinates.csv")
        if not os.path.exists(coordinates_file):
            return False
            
        return current_mtime == initial_mtime
    except Exception as e:
        print(f"Error checking folder stability: {e}")
        return False


async def setup_hypha_connection() -> bool:
    """Set up the Hypha connection and initialize managers."""
    global gallery_manager, uploader
    
    try:
        # Initialize gallery manager with connection
        gallery_manager = GalleryManager()
        await gallery_manager.ensure_connected()
        
        # Initialize uploader with connection
        uploader = ArtifactUploader(
            artifact_alias="",  # Will be set per dataset
            client_id=client_id
        )
        await uploader.connect_with_retry()
        
        return True
    except Exception as e:
        print(f"Error setting up Hypha connection: {e}")
        return False


async def create_gallery_if_not_exists() -> bool:
    """Create the gallery if it doesn't exist."""
    global gallery_manager
    
    try:
        await gallery_manager.create_gallery(
            name=GALLERY_NAME,
            description=GALLERY_DESCRIPTION,
            alias=GALLERY_ALIAS,
            permissions={"*": "*", "@": "*", "misty-teeth-42051243": "*", "google-oauth2|103047988474094226050": "*"}
        )
        return True
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Gallery {GALLERY_ALIAS} already exists, continuing...")
            return True
        print(f"Error creating gallery: {e}")
        return False


async def create_dataset_for_folder(folder_name: str) -> Optional[str]:
    """Create a dataset for a specific folder timestamp."""
    global gallery_manager
    
    folder_datetime = extract_datetime_from_folder(folder_name)
    if not folder_datetime:
        print(f"Could not extract datetime from folder name: {folder_name}")
        return None
        
    dataset_alias = f"agent-lens/{EXPERIMENT_ID}-{folder_datetime}"
    dataset_name = f"Image Map {folder_datetime}"
    dataset_description = f"The Image Map from {folder_datetime}"
    
    try:
        await gallery_manager.create_dataset(
            name=dataset_name,
            description=dataset_description,
            alias=dataset_alias,
            parent_id=GALLERY_ALIAS,
            version="stage",
            permissions={"*": "*", "@": "*", "misty-teeth-42051243": "*", "google-oauth2|103047988474094226050": "*"}
        )
        return dataset_alias
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Dataset {dataset_alias} already exists, continuing...")
            return dataset_alias
        print(f"Error creating dataset: {e}")
        return None


async def stitch_folder(folder_path: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
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
            return False, None, None
        if not os.path.exists(parameter_file):
            print(f"Parameter file {parameter_file} does not exist")
            return False, None, None
            
        # Load coordinates file
        if os.path.exists(coordinates_file):
            coordinates = pd.read_csv(coordinates_file)
        else:
            print(f"Coordinates file {coordinates_file} not found")
            return False, None, None
            
        # Stage limits
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
            return False, None, None
            
        channels = list(set(info["channel_name"] for info in image_info))
        print(f"Found {len(image_info)} images with {len(channels)} channels")
        selected_channel = ['Fluorescence_561_nm_Ex', 'Fluorescence_488_nm_Ex', 'BF_LED_matrix_full']
        # Filter selected channels to only include ones that are present
        selected_channel = [c for c in selected_channel if c in channels]
        print(f"Selected channels: {selected_channel}")
        
        if not selected_channel:
            print("No matching channels found in the data")
            return False, None, None
        
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
        
        # Process images in background thread
        await asyncio.to_thread(
            stitch_manager.process_images,
            image_info=image_info,
            coordinates=coordinates,
            selected_channel=selected_channel,
            pyramid_levels=6
        )
        
        # Generate chunk metadata
        chunk_metadata = {}
        
        # Extract imaging parameters for metadata
        with open(parameter_file, 'r') as f:
            imaging_params = json.load(f)
        
        # Create metadata for each coordinate in the stitched image
        for _, row in coordinates.iterrows():
            region = row.get('region', 'unknown')
            x_idx = int(row.get('i', 0))
            y_idx = int(row.get('j', 0))
            
            # Physical coordinates
            x_mm = float(row.get('x (mm)', 0))
            y_mm = float(row.get('y (mm)', 0))
            z_mm = float(row.get('z (mm)', 0))
            
            # Create a chunk identifier string based on the x,y,z indices
            # Default to (0, z-idx, y-idx, x-idx) since zarr default dimension order is t,z,y,x
            # For simplicity, we're using (y-idx, x-idx) as our 2D key
            chunk_key = f"({y_idx}, {x_idx})"
            
            # Collect metadata for this chunk
            chunk_metadata[chunk_key] = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "folder_name": folder_name,
                "region": region,
                "coordinates_grid": (x_idx, y_idx),  # Grid coordinates (i, j)
                "location_mm": (x_mm, y_mm, z_mm),   # Physical stage position in mm
                "acquisition_params": {
                    "exposure_ms": imaging_params.get("exposure_ms", 0),
                    "gain": imaging_params.get("gain", 0)
                }
            }
        
        zarr_path = os.path.join(STITCHED_DIR, zarr_filename)
        
        # Apply the chunk metadata to the zarr file
        print("Adding chunk metadata to zarr file...")
        metadata_success = add_chunk_metadata(zarr_path, chunk_metadata)
        if not metadata_success:
            print("Warning: Failed to add chunk metadata")
        
        # Add dataset-level metadata
        dataset_metadata = {
            "title": f"Image Map {folder_datetime}",
            "description": f"Stitched image map from {folder_name}",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "software_version": "1.0.0",
            "experiment_id": EXPERIMENT_ID,
            "imaging_modality": "brightfield and fluorescence",
            "channels": selected_channel
        }
        
        # We can use the general dataset metadata function too
        stitch_manager.add_dataset_metadata(zarr_path, dataset_metadata)
        
        print(f"Stitching successfully completed, zarr file created: {zarr_filename}")
        return True, zarr_filename, chunk_metadata
        
    except Exception as e:
        print(f"Error during stitching: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


# Function to run in separate process for isolated network context
def isolated_upload_process(zarr_path, channel, artifact_alias, client_id, result_queue):
    """Run the upload in a separate process to isolate network context"""
    try:
        # Set up a new asyncio event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Create a new uploader instance in this process
        uploader = ArtifactUploader(
            artifact_alias=artifact_alias,
            client_id=f"{client_id}-subprocess-{os.getpid()}"
        )
        
        # Execute the upload
        success = loop.run_until_complete(process_channel_upload(zarr_path, channel, uploader))
        
        # Put result in queue for parent process
        result_queue.put({"success": success, "channel": channel})
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        result_queue.put({"success": False, "channel": channel, "error": str(e), "traceback": error_details})
    finally:
        # Clean up loop
        loop.close()

async def process_channel_upload(zarr_path, channel, uploader):
    """Process the actual upload in the isolated process"""
    # Get the base name for the relative path
    zarr_file = os.path.basename(zarr_path)
    base_name = zarr_file.replace('.zarr', '')
    
    # The full path to the channel in the zarr file
    channel_path = os.path.join(zarr_path, channel)
    
    if not os.path.exists(channel_path):
        print(f"Channel directory {channel_path} not found in zarr file")
        return False
    
    # Set the relative path for upload
    relative_path = f"{base_name}/{channel}.zip"
    print(f"Uploading channel {channel} as {relative_path} in isolated process {os.getpid()}")
    
    # First establish connection
    connect_success = await uploader.connect_with_retry()
    if not connect_success:
        print(f"Failed to establish connection to Hypha in isolated process")
        return False
    
    # Use the zip_and_upload_folder method
    try:
        success = await uploader.zip_and_upload_folder(
            folder_path=channel_path,
            relative_path=relative_path,
            delete_zip_after=True
        )
        
        if success:
            print(f"Successfully uploaded channel {channel} in isolated process")
        else:
            print(f"Failed to upload channel {channel} in isolated process")
        
        return success
    finally:
        # Always disconnect at the end
        if uploader.connection:
            await uploader.connection.disconnect()

# Replace the existing upload_zarr_channel function with one that uses multiprocessing
async def upload_zarr_channel(zarr_path: str, channel_name: str, dataset_alias: str) -> bool:
    """
    Uploads contents of a Zarr channel group.
    Specifically, for each scale array within the channel:
    - Uploads scale's .zarray and .zattrs files.
    - Zips and uploads each first-level chunk directory within the scale array.
    Example upload paths:
    - channel_name/scale_name/.zarray
    - channel_name/scale_name/chunk_dir_name.zip
    """
    global uploader # This uploader instance is initialized in setup_hypha_connection and configured by process_folder
    # uploader.artifact_alias should already be set by the caller (e.g., process_folder) to dataset_alias

    channel_group_local_path = os.path.join(zarr_path, channel_name)

    if not os.path.isdir(channel_group_local_path):
        print(f"Channel group path {channel_group_local_path} not found for channel {channel_name}.")
        return False

    # Iterate over scale arrays within the channel group (e.g., scale0, scale1)
    for scale_array_name in os.listdir(channel_group_local_path): # e.g. "scale0", "scale1"
        scale_array_local_path = os.path.join(channel_group_local_path, scale_array_name)

        # Check if it's a Zarr array (must be a directory and contain .zarray)
        if os.path.isdir(scale_array_local_path) and \
           os.path.exists(os.path.join(scale_array_local_path, ".zarray")):
            
            print(f"    Processing scale array: {channel_name}/{scale_array_name}")

            # Upload metadata files (.zarray, .zattrs, etc.) for this scale array
            # Also, zip and upload chunk group directories
            for item_in_scale_array in os.listdir(scale_array_local_path):
                item_local_path = os.path.join(scale_array_local_path, item_in_scale_array)

                if os.path.isfile(item_local_path):
                    # Typically .zarray, .zattrs. Could be other metadata files.
                    relative_upload_path = f"{channel_name}/{scale_array_name}/{item_in_scale_array}"
                    print(f"      Uploading metadata file: {relative_upload_path}")
                    success = await uploader.upload_single_file(
                        local_file=item_local_path,
                        relative_path=relative_upload_path
                    )
                    if not success:
                        print(f"      Failed to upload metadata file {relative_upload_path}")
                        return False
                
                elif os.path.isdir(item_local_path):
                    # This is a chunk group directory within the scale array (e.g., "0", "1" for first dimension of chunks)
                    chunk_group_dir_to_zip = item_local_path
                    chunk_group_name = item_in_scale_array # e.g., "0"
                    
                    # Relative path for the zip file: channel_name/scale_array_name/chunk_group_name.zip
                    zip_relative_upload_path = f"{channel_name}/{scale_array_name}/{chunk_group_name}.zip"
                    
                    print(f"      Zipping and uploading chunk group: {zip_relative_upload_path}")
                    success = await uploader.zip_and_upload_folder(
                        folder_path=chunk_group_dir_to_zip,
                        relative_path=zip_relative_upload_path,
                        delete_zip_after=True
                    )
                    if not success:
                        print(f"      Failed to upload chunk group zip {zip_relative_upload_path}")
                        return False
            # Finished processing items in this scale_array_local_path
        # else:
            # This item within the channel group is not a Zarr array recognized by the presence of .zarray.
            # It could be the .zgroup or .zattrs file for the channel group itself, which is handled by the calling function (process_folder).
            pass
            
    return True # All processed successfully for this channel


async def upload_zarr_file(zarr_file: str) -> bool:
    """Upload a zarr file to the artifact manager by zipping and uploading each channel separately."""
    zarr_path = os.path.join(STITCHED_DIR, zarr_file)
    if not os.path.exists(zarr_path):
        print(f"Zarr file {zarr_path} does not exist")
        return False
    
    # Create uploader instance with optimized concurrency
    uploader = ArtifactUploader(
        artifact_alias=ARTIFACT_ALIAS,
        client_id=client_id
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

        # Put dataset in staging mode
        staging_success = False
        retry_count = 0
        
        while not staging_success and retry_count < MAX_RETRIES:
            try:
                print(f"Putting dataset {ARTIFACT_ALIAS} in staging mode...")
                # Use the artifact_manager from the uploader's connection
                artifact_manager = connection.artifact_manager 
                if not artifact_manager:
                    raise ValueError("Artifact manager is not available on the connection.")

                # Read the current manifest with timeout (if needed, otherwise just edit)
                dataset_manifest = {
                    "name": "Full zarr dataset 20250429",
                    "description": "The Full zarr dataset for U2OS FUCCI Drug Treatment from 20250429",
                }
                
                # Put the dataset in staging mode with timeout
                await asyncio.wait_for(
                    artifact_manager.edit(
                        artifact_id=ARTIFACT_ALIAS,
                        manifest=dataset_manifest,  # Preserve the same manifest
                        version="stage"    # Put in staging mode
                    ),
                    timeout=OPERATION_TIMEOUT
                )
                
                print("Dataset is now in staging mode")
                staging_success = True
                
            except asyncio.TimeoutError:
                retry_count += 1
                print(f"Staging operation timed out (attempt {retry_count}/{MAX_RETRIES})")
                
                # Reset connection using the uploader's method
                print("Attempting to reset connection via uploader...")
                await connection.disconnect() # Disconnect first
                if retry_count < MAX_RETRIES:
                    connect_success = await uploader.connect_with_retry(client_id=client_id)
                    if not connect_success:
                        print("Failed to re-establish connection after timeout.")
                        return False
                    # Update connection object reference after successful reconnect
                    connection = uploader.connection 
                    if not connection or not connection.artifact_manager:
                        print("Failed to get valid connection after reconnect.")
                        return False
                    await asyncio.sleep(5) # Wait a bit after reconnecting
            
            except Exception as e:
                retry_count += 1
                print(f"Error putting dataset in staging mode: {e} (attempt {retry_count}/{MAX_RETRIES})")
                
                if "not found" in str(e).lower():
                    print("Dataset not found. It may need to be created first.")
                    # Disconnect cleanly before returning
                    if connection: await connection.disconnect()
                    return False
                
                # Reset connection using the uploader's method
                print("Attempting to reset connection via uploader due to error...")
                if connection: await connection.disconnect() # Disconnect first
                if retry_count < MAX_RETRIES:
                    connect_success = await uploader.connect_with_retry(client_id=client_id)
                    if not connect_success:
                        print("Failed to re-establish connection after error.")
                        return False
                     # Update connection object reference after successful reconnect
                    connection = uploader.connection
                    if not connection or not connection.artifact_manager:
                         print("Failed to get valid connection after error reconnect.")
                         return False
                    await asyncio.sleep(5) # Wait a bit after reconnecting
        
        if not staging_success:
            print("Failed to put dataset in staging mode after multiple attempts")
            if connection: await connection.disconnect()
            return False
        
        # Get the channels from the zarr file (top-level directories in zarr)
        channels = [d for d in os.listdir(zarr_path) if os.path.isdir(os.path.join(zarr_path, d))]
        print(f"Found channels in zarr file: {channels}")
        
        # Upload each channel separately as its own zip file
        all_success = True
        for channel in channels:
            print(f"Processing channel: {channel}")
            channel_success = await upload_zarr_channel(zarr_path, channel, ARTIFACT_ALIAS)
            if not channel_success:
                print(f"Failed to upload channel {channel}")
                all_success = False
                break
        
        if not all_success:
            print("Failed to upload all channels")
            if connection: await connection.disconnect()
            return False
        
        # Commit the dataset
        commit_success = False
        commit_attempts = 0
        
        while not commit_success and commit_attempts < Config.MAX_COMMIT_ATTEMPTS:
            try:
                # Commit the dataset with timeout
                print(f"Committing dataset {ARTIFACT_ALIAS}...")
                await asyncio.wait_for(
                    connection.artifact_manager.commit(ARTIFACT_ALIAS),
                    timeout=Config.MAX_COMMIT_DELAY
                )
                print("Dataset committed successfully.")
                commit_success = True
                
            except asyncio.TimeoutError:
                commit_attempts += 1
                print(f"Commit operation timed out (attempt {commit_attempts}/{Config.MAX_COMMIT_ATTEMPTS})")
                if connection: await connection.disconnect()
                await asyncio.sleep(min(Config.INITIAL_RETRY_DELAY * (2 ** commit_attempts), Config.MAX_RETRY_DELAY))
                
            except Exception as e:
                commit_attempts += 1
                print(f"Error committing dataset (attempt {commit_attempts}/{Config.MAX_COMMIT_ATTEMPTS}): {str(e)}")
                if connection: await connection.disconnect()
                await asyncio.sleep(min(Config.INITIAL_RETRY_DELAY * (2 ** commit_attempts), Config.MAX_RETRY_DELAY))
        
        if not commit_success:
            print(f"WARNING: Failed to commit the dataset {ARTIFACT_ALIAS} after {Config.MAX_COMMIT_ATTEMPTS} attempts.")
            if connection: await connection.disconnect()
            return False
            
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
    """Process a single folder: stitch and upload."""
    global gallery_manager, uploader
    
    folder_path = os.path.join(BASE_DIR, folder_name)
    
    # Wait for folder to be stable
    print(f"Waiting for folder {folder_name} to be stable...")
    while not await is_folder_stable(folder_path):
        print(f"Folder {folder_name} is still being modified, waiting...")
        await asyncio.sleep(CHECK_INTERVAL)
    
    # Create dataset for this folder
    dataset_alias = await create_dataset_for_folder(folder_name)
    if not dataset_alias:
        print(f"Failed to create dataset for folder {folder_name}")
        return False
    
    # Stitch the folder
    print(f"\nStarting stitching process for folder: {folder_name}")
    stitch_success, zarr_filename, chunk_metadata = await stitch_folder(folder_path)
    if not stitch_success:
        print(f"Failed to stitch folder: {folder_name}")
        return False
    
    # Upload the zarr file
    zarr_path = os.path.join(STITCHED_DIR, zarr_filename)
    if not os.path.exists(zarr_path):
        print(f"Zarr file {zarr_path} does not exist")
        return False
    
    try:
        # Ensure uploader is set for this dataset_alias (created by create_dataset_for_folder)
        uploader.artifact_alias = dataset_alias

        # 1. Upload root .zgroup for the dataset
        root_zgroup_local_path = os.path.join(zarr_path, ".zgroup")
        if os.path.exists(root_zgroup_local_path):
            print(f"Uploading root .zgroup for dataset {dataset_alias}")
            success = await uploader.upload_single_file(
                local_file=root_zgroup_local_path,
                relative_path=".zgroup" # Relative to dataset_alias
            )
            if not success:
                print(f"Failed to upload root .zgroup for {dataset_alias}")
                return False
        else:
            print(f"Warning: Root .zgroup not found at {root_zgroup_local_path}")

        # Get channels from the zarr file (these are Zarr groups)
        channel_names = []
        if os.path.exists(zarr_path): # Ensure zarr_path exists before listing its contents
            channel_names = [d for d in os.listdir(zarr_path) if os.path.isdir(os.path.join(zarr_path, d))]
        
        print(f"Found channels in zarr file {zarr_filename}: {channel_names}")
        
        all_uploads_successful = True
        for channel_name in channel_names:
            print(f"Processing channel group: {channel_name} for dataset {dataset_alias}")

            # 2a. Upload .zgroup for the channel group itself
            channel_zgroup_local_path = os.path.join(zarr_path, channel_name, ".zgroup")
            if os.path.exists(channel_zgroup_local_path):
                relative_channel_zgroup_path = f"{channel_name}/.zgroup"
                print(f"  Uploading .zgroup for channel {channel_name} to {relative_channel_zgroup_path}")
                success = await uploader.upload_single_file(
                    local_file=channel_zgroup_local_path,
                    relative_path=relative_channel_zgroup_path
                )
                if not success:
                    print(f"  Failed to upload .zgroup for channel {channel_name}")
                    all_uploads_successful = False; break
            else:
                print(f"  Warning: .zgroup not found for channel {channel_name} at {channel_zgroup_local_path}")
            
            # 2a. Upload .zattrs for the channel group itself (if exists)
            channel_zattrs_local_path = os.path.join(zarr_path, channel_name, ".zattrs")
            if os.path.exists(channel_zattrs_local_path):
                relative_channel_zattrs_path = f"{channel_name}/.zattrs"
                print(f"  Uploading .zattrs for channel {channel_name} to {relative_channel_zattrs_path}")
                success = await uploader.upload_single_file(
                    local_file=channel_zattrs_local_path,
                    relative_path=relative_channel_zattrs_path
                )
                if not success:
                    print(f"  Failed to upload .zattrs for channel {channel_name}")
                    all_uploads_successful = False; break
            # else: Not all channel groups will have a .zattrs, so this is not necessarily a warning

            if not all_uploads_successful: break

            # 2b. Call specialized function to upload contents of this channel group (scales, chunks)
            print(f"  Uploading data for channel {channel_name}...")
            channel_content_success = await upload_zarr_channel(zarr_path, channel_name, dataset_alias)
            if not channel_content_success:
                print(f"  Failed to upload data contents for channel {channel_name}")
                all_uploads_successful = False; break
        
        if not all_uploads_successful:
            print(f"Failed to upload all parts of Zarr dataset {dataset_alias} for folder {folder_name}")
            return False
        
        # Commit the dataset
        print(f"All uploads for {dataset_alias} successful. Committing dataset...")
        commit_success = await gallery_manager.commit_dataset(dataset_alias)
        if not commit_success:
            print(f"Failed to commit dataset {dataset_alias}")
            return False
        
        # Clean up the specific zarr directory for this folder
        print(f"Cleaning up zarr path: {zarr_path}")
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        return True
        
    except Exception as e:
        print(f"Error during folder processing {folder_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    # Set up Hypha connection as an asyncio task
    connection_task = asyncio.create_task(setup_hypha_connection())
    
    try:
        # Wait for connection to be established
        if not await connection_task:
            print("Failed to set up Hypha connection, exiting...")
            return
        
        # Create gallery if it doesn't exist
        if not await create_gallery_if_not_exists():
            print("Failed to create gallery, exiting...")
            return
        
        while True:  # Main monitoring loop
            # Get all folders
            all_folders = get_timelapse_folders()
            if not all_folders:
                print(f"No folders matching {EXPERIMENT_ID} found in {BASE_DIR}")
                await asyncio.sleep(CHECK_INTERVAL)
                continue
            
            processed_folders = get_processed_folders()
            
            print("\nAvailable folders:")
            for i, folder in enumerate(all_folders):
                status = " (processed)" if folder in processed_folders else ""
                print(f"{i+1}: {folder}{status}")
            
            # Ask user which folder to start with
            user_input = input("\nWhich folder do you want to start with? (enter number or folder name): ")
            
            # Parse start folder input
            if user_input.isdigit() and 1 <= int(user_input) <= len(all_folders):
                start_folder = all_folders[int(user_input) - 1]
            elif user_input in all_folders:
                start_folder = user_input
            else:
                print(f"Invalid input. Using first unprocessed folder.")
                unprocessed = [f for f in all_folders if f not in processed_folders]
                if not unprocessed:
                    print("No unprocessed folders found.")
                    await asyncio.sleep(CHECK_INTERVAL)
                    continue
                start_folder = unprocessed[0]
            
            # Ask user which folder to end with
            end_input = input("\nWhich folder do you want to end with? (enter number, folder name, or press Enter for continuous monitoring): ")
            
            # Parse end folder input
            end_folder = None
            if end_input:
                if end_input.isdigit() and 1 <= int(end_input) <= len(all_folders):
                    end_folder = all_folders[int(end_input) - 1]
                elif end_input in all_folders:
                    end_folder = end_input
                else:
                    print(f"Invalid end folder input. Will process until the end of the list.")
            
            # Find the index of the start folder
            start_idx = all_folders.index(start_folder)
            
            # Process folders starting from the selected one
            current_idx = start_idx
            while current_idx < len(all_folders):
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
                
                # Check if we've reached the end folder AFTER processing it
                if end_folder and folder_to_process == end_folder:
                    print(f"\nReached end folder: {end_folder}")
                    if input("\nDo you want to continue monitoring for new folders? (y/n): ").lower() != 'y':
                        return
                    break
            
            # If no end folder was specified, continue monitoring
            if not end_folder:
                print("\nWaiting for new folders...")
                await asyncio.sleep(CHECK_INTERVAL)
    
    finally:
        # Clean up connections
        if gallery_manager and gallery_manager.connection:
            await gallery_manager.connection.disconnect()
        if uploader and uploader.connection:
            await uploader.connection.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Clean up temporary directory
        if os.path.exists(STITCHED_DIR):
            shutil.rmtree(STITCHED_DIR) 