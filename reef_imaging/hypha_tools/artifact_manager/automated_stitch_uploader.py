import os
import sys
import time
import asyncio
import aiohttp
import json
import shutil
import subprocess
import re
from datetime import datetime
from typing import List, Optional, Tuple
from dotenv import load_dotenv
import cv2
import numpy as np
import pandas as pd
import zarr

# Constants
BASE_DIR = "/media/reef/harddisk"
EXPERIMENT_ID = "20250410-fucci-time-lapse-scan"
STITCHED_DIR = "/media/reef/harddisk/test_stitch_zarr"
STITCH_RECORD_FILE = "stitch_upload_progress.txt"
UPLOAD_RECORD_FILE = "zarr_upload_record.json"
CHECK_INTERVAL = 60  # Check for new folders every 60 seconds

# Load environment variables
load_dotenv()
SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_TOKEN = os.getenv("REEF_WORKSPACE_TOKEN")
ARTIFACT_ALIAS = "image-map-20250410-treatment"

# Connection settings
CONCURRENCY_LIMIT = 5  # Max number of concurrent uploads
MAX_RETRIES = 300  # Maximum number of retry attempts
INITIAL_RETRY_DELAY = 5  # Initial retry delay in seconds
MAX_RETRY_DELAY = 60  # Maximum retry delay in seconds
CONNECTION_TIMEOUT = 30  # Timeout for API connections in seconds
UPLOAD_TIMEOUT = 120  # Timeout for file uploads in seconds


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


def load_upload_record():
    """Load the record of previously uploaded files"""
    if os.path.exists(UPLOAD_RECORD_FILE):
        with open(UPLOAD_RECORD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert uploaded_files from list to set if needed
            if isinstance(data["uploaded_files"], list):
                data["uploaded_files"] = set(data["uploaded_files"])
            return data
    return {
        "uploaded_files": set(),
        "last_update": None,
        "total_files": 0,
        "completed_files": 0,
    }


def save_upload_record(record):
    """Save the record of uploaded files"""
    # Convert set to list for JSON serialization
    record_copy = record.copy()
    record_copy["uploaded_files"] = list(record["uploaded_files"])
    record_copy["last_update"] = datetime.now().isoformat()

    with open(UPLOAD_RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(record_copy, f, indent=2)


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
    """Implement stitching functionality directly instead of calling stitch_zarr.py"""
    # Clean up existing zarr files first
    print(f"Cleaning up any existing zarr files in {STITCHED_DIR}")
    existing_zarr_files = get_zarr_files()
    for zarr_file in existing_zarr_files:
        cleanup_zarr_file(zarr_file)
        
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
            
        # Load imaging parameters
        try:
            with open(parameter_file, "r") as f:
                parameters = json.load(f)
        except Exception as e:
            print(f"Error loading parameters file: {e}")
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
        
        # Calculate pixel size from parameters
        def get_pixel_size(parameters):
            try:
                tube_lens_mm = float(parameters.get('tube_lens_mm', 50.0))
                pixel_size_um = float(parameters.get('sensor_pixel_size_um', 1.85))
                objective_tube_lens_mm = float(parameters['objective'].get('tube_lens_f_mm', 180.0))
                magnification = float(parameters['objective'].get('magnification', 20.0))
                adjustment_factor = 0.936  # Manual adjustment factor
            except KeyError:
                print("Warning: Missing parameters for pixel size calculation, using defaults")
                tube_lens_mm = 50.0
                pixel_size_um = 1.85
                objective_tube_lens_mm = 180.0
                magnification = 20.0
                adjustment_factor = 0.936
                
            pixel_size_xy = pixel_size_um / (magnification / (objective_tube_lens_mm / tube_lens_mm))
            pixel_size_xy *= adjustment_factor
            print(f"Pixel size: {pixel_size_xy} µm (with adjustment factor: {adjustment_factor})")
            return pixel_size_xy
            
        # Calculate canvas size
        def create_canvas_size(stage_limits, pixel_size_xy):
            x_range = (stage_limits["x_positive"] - stage_limits["x_negative"]) * 1000  # Convert mm to µm
            y_range = (stage_limits["y_positive"] - stage_limits["y_negative"]) * 1000  # Convert mm to µm
            canvas_width = int(x_range / pixel_size_xy)
            canvas_height = int(y_range / pixel_size_xy)
            return canvas_width, canvas_height
            
        # Parse image filenames
        def parse_image_filenames(image_folder):
            image_files = [f for f in os.listdir(image_folder) if f.endswith(".bmp")]
            image_info = []

            for image_file in image_files:
                # Split only for the first 4 parts (region, x, y, z)
                prefix_parts = image_file.split('_', 4)  # Split into max 5 parts
                if len(prefix_parts) >= 5:
                    region, x_idx, y_idx, z_idx = prefix_parts[:4]
                    # Get the channel name by removing the extension
                    channel_name = prefix_parts[4].rsplit('.', 1)[0]

                    image_info.append({
                        "filepath": os.path.join(image_folder, image_file),
                        "region": region,
                        "x_idx": int(x_idx),
                        "y_idx": int(y_idx),
                        "z_idx": z_idx,
                        "channel_name": channel_name
                    })

            # Sort images according to the grid pattern
            image_info.sort(key=lambda x: (-x["x_idx"], x["y_idx"]))
            return image_info
            
        # Rotate and flip image function
        def rotate_flip_image(image, angle=0, flip=True):
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
            if flip:
                rotated = cv2.flip(rotated, -1)  # Flip horizontally and vertically
            return rotated
            
        # Create zarr file with pyramid
        def create_ome_ngff(output_folder, canvas_width, canvas_height, channels, zarr_filename, pyramid_levels=7, chunk_size=(2048, 2048)):
            os.makedirs(output_folder, exist_ok=True)
            zarr_path = os.path.join(output_folder, zarr_filename)
            
            store = zarr.DirectoryStore(zarr_path)
            root = zarr.group(store=store)

            print(f"Initial canvas size: {canvas_width}x{canvas_height}")

            # Create datasets for each channel
            datasets = {}
            for channel in channels:
                print(f"\nCreating dataset for channel: {channel}")
                group = root.create_group(channel)
                datasets[channel] = group

                # Create base resolution (scale0)
                print(f"scale0: {canvas_height}x{canvas_width}")
                datasets[channel].create_dataset(
                    "scale0",
                    shape=(canvas_height, canvas_width),
                    chunks=chunk_size,
                    dtype=np.uint8,
                    compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)
                )

                # Create pyramid levels
                for level in range(1, pyramid_levels):
                    scale_name = f"scale{level}"
                    level_shape = (
                        max(1, canvas_height // (4 ** level)),
                        max(1, canvas_width // (4 ** level))
                    )
                    level_chunks = (
                        min(chunk_size[0], level_shape[0]),
                        min(chunk_size[1], level_shape[1])
                    )

                    print(f"{scale_name}: {level_shape} (chunks: {level_chunks})")

                    datasets[channel].create_dataset(
                        scale_name,
                        shape=level_shape,
                        chunks=level_chunks,
                        dtype=np.uint8,
                        compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)
                    )

            return root, datasets
            
        # Update pyramid levels
        def update_pyramid(datasets, channel, level, image, x_start, y_start):
            scale = 4 ** level
            scale_name = f"scale{level}"

            try:
                pyramid_dataset = datasets[channel][scale_name]

                # Calculate the scaled coordinates
                x_scaled = x_start // scale
                y_scaled = y_start // scale

                # Calculate target dimensions
                target_height = max(1, image.shape[0] // scale)
                target_width = max(1, image.shape[1] // scale)

                # Only proceed if the target dimensions are meaningful
                if target_height == 0 or target_width == 0:
                    return

                # Resize the image
                scaled_image = cv2.resize(
                    image, 
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA
                )

                # Calculate the valid region to update
                end_y = min(y_scaled + scaled_image.shape[0], pyramid_dataset.shape[0])
                end_x = min(x_scaled + scaled_image.shape[1], pyramid_dataset.shape[1])

                # Only update if we have a valid region
                if end_y > y_scaled and end_x > x_scaled:
                    # Clip the image if necessary
                    if end_y - y_scaled != scaled_image.shape[0] or end_x - x_scaled != scaled_image.shape[1]:
                        scaled_image = scaled_image[:(end_y - y_scaled), :(end_x - x_scaled)]

                    # Place the image
                    pyramid_dataset[y_scaled:end_y, x_scaled:end_x] = scaled_image

            except Exception as e:
                print(f"Error updating pyramid level {level}: {e}")
            finally:
                # Clean up
                if 'scaled_image' in locals():
                    del scaled_image
        
        # Process and stitch images
        def process_images(image_info, coordinates, datasets, pixel_size_xy, stage_limits, pyramid_levels=7, selected_channel=None):
            if selected_channel is None:
                raise ValueError("selected_channel must be a list of channels to process.")

            # Convert stage limits from mm to μm for offset calculation
            x_offset = stage_limits["x_negative"] * 1000  # Convert mm to µm
            y_offset = stage_limits["y_negative"] * 1000  # Convert mm to µm

            # Get canvas dimensions from the dataset
            canvas_width = datasets[selected_channel[0]]["scale0"].shape[1]
            canvas_height = datasets[selected_channel[0]]["scale0"].shape[0]
            
            # Process images by position (same x,y,region) together
            position_groups = {}
            
            for info in image_info:
                position_key = (info["region"], info["x_idx"], info["y_idx"])
                if position_key not in position_groups:
                    position_groups[position_key] = []
                position_groups[position_key].append(info)
            
            # Sort position keys
            sorted_positions = sorted(position_groups.keys(), key=lambda pos: (
                pos[0],  # region
                -pos[1],  # i decreasing
                pos[2]    # j ascending
            ))
            
            # Find the first image to determine tile dimensions
            sample_image = None
            for position in sorted_positions:
                if len(position_groups[position]) > 0:
                    sample_info = position_groups[position][0]
                    sample_path = sample_info["filepath"]
                    sample_image = cv2.imread(sample_path, cv2.IMREAD_ANYDEPTH)
                    if sample_image is not None:
                        sample_image = rotate_flip_image(sample_image)
                        break
            
            if sample_image is None:
                raise ValueError("Could not find any valid images to process")
            
            tile_height, tile_width = sample_image.shape[:2]
            print(f"Tile dimensions: {tile_width}x{tile_height}")
            
            # Process all positions
            for position in sorted_positions:
                region, x_idx, y_idx = position

                # Get coordinates for this position
                matching_rows = coordinates[(coordinates["region"] == region) & 
                                          (coordinates["i"] == x_idx) & 
                                          (coordinates["j"] == y_idx)]
                if matching_rows.empty:
                    print(f"Warning: No coordinates found for position {position}")
                    continue
                    
                coord_row = matching_rows.iloc[0]
                x_mm, y_mm = coord_row["x (mm)"], coord_row["y (mm)"]
                
                # Convert mm to μm, then to pixels
                x_um = (x_mm * 1000) - x_offset
                y_um = (y_mm * 1000) - y_offset
                x_start = int(x_um / pixel_size_xy)
                y_start = int(y_um / pixel_size_xy)

                # Process all channels at this position
                for info in position_groups[position]:
                    channel = info["channel_name"]
                    if channel not in selected_channel:
                        continue

                    # Read and preprocess image
                    image = cv2.imread(info["filepath"], cv2.IMREAD_ANYDEPTH)
                    if image is None:
                        print(f"Error: Unable to read {info['filepath']}")
                        continue

                    # Normalize image to 8-bit
                    if image.dtype != np.uint8:
                        img_min = np.percentile(image, 1)
                        img_max = np.percentile(image, 99)
                        image = np.clip(image, img_min, img_max)
                        image = ((image - img_min) * 255 / (img_max - img_min)).astype(np.uint8)
                    
                    # Rotate and flip
                    image = rotate_flip_image(image)
                    
                    # Check if image dimensions match the expected tile size
                    if image.shape[0] != tile_height or image.shape[1] != tile_width:
                        # Create a new empty image with the expected tile size
                        temp_image = np.zeros((tile_height, tile_width), dtype=np.uint8)
                        # Copy as much of the original image as fits
                        h = min(image.shape[0], tile_height)
                        w = min(image.shape[1], tile_width)
                        temp_image[:h, :w] = image[:h, :w]
                        image = temp_image

                    # Validate and clip the placement region
                    if x_start < 0 or y_start < 0 or x_start >= canvas_width or y_start >= canvas_height:
                        print(f"Warning: Image at {info['filepath']} is out of canvas bounds.")
                        continue

                    # Calculate end positions and clip image if necessary
                    x_end = min(x_start + image.shape[1], canvas_width)
                    y_end = min(y_start + image.shape[0], canvas_height)

                    # Create a trimmed image if it exceeds canvas boundaries
                    if x_end - x_start != image.shape[1] or y_end - y_start != image.shape[0]:
                        trimmed_image = image[:(y_end-y_start), :(x_end-x_start)]
                    else:
                        trimmed_image = image

                    # Place image directly on canvas at base level (scale0)
                    datasets[channel]["scale0"][y_start:y_end, x_start:x_end] = trimmed_image

                    # Update pyramid levels
                    for level in range(1, pyramid_levels):
                        update_pyramid(datasets, channel, level, image, x_start, y_start)

                    del image
        
        # Execute stitching process
        pixel_size_xy = get_pixel_size(parameters)
        canvas_width, canvas_height = create_canvas_size(stage_limits, pixel_size_xy)
        
        # Calculate appropriate number of pyramid levels
        max_dimension = max(canvas_width, canvas_height)
        max_levels = int(np.floor(np.log2(max_dimension)) // 2)  # Divide by 2 since we use 4^level
        pyramid_levels = min(6, max_levels) if max_levels > 0 else 1
        print(f"Using {pyramid_levels} pyramid levels for {canvas_width}x{canvas_height} canvas")
        
        # Parse image filenames and get unique channels
        image_info = parse_image_filenames(image_folder)
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
        
        # Create new OME-NGFF file
        root, datasets = create_ome_ngff(STITCHED_DIR, canvas_width, canvas_height, 
                                         selected_channel, zarr_filename=zarr_filename, 
                                         pyramid_levels=pyramid_levels)
        
        # Process images and stitch them
        process_images(image_info, coordinates, datasets, pixel_size_xy, stage_limits, 
                      selected_channel=selected_channel, pyramid_levels=pyramid_levels)
        
        print(f"Stitching successfully completed, zarr file created: {zarr_filename}")
        return True
    except Exception as e:
        print(f"Error during stitching: {e}")
        import traceback
        traceback.print_exc()
        return False


async def get_artifact_manager(timeout=CONNECTION_TIMEOUT):
    """Get a new connection to the artifact manager with timeout"""
    try:
        from hypha_rpc import connect_to_server
        api = await asyncio.wait_for(
            connect_to_server({
                "name": "test-client", 
                "server_url": SERVER_URL, 
                "token": WORKSPACE_TOKEN
            }),
            timeout=timeout
        )
        artifact_manager = await asyncio.wait_for(
            api.get_service("public/artifact-manager"),
            timeout=timeout
        )
        return api, artifact_manager
    except asyncio.TimeoutError:
        print(f"Connection timed out after {timeout} seconds")
        raise
    except Exception as e:
        print(f"Error connecting to artifact manager: {e}")
        raise


async def upload_single_file(
    artifact_manager,
    artifact_alias,
    local_file,
    relative_path,
    semaphore,
    session,
    upload_record,
):
    """
    Single attempt to upload a file. No internal retries.
    """
    # Skip if file was already uploaded
    if relative_path in upload_record["uploaded_files"]:
        print(f"Skipping already uploaded file: {relative_path}")
        return True

    retry_count = 0
    retry_delay = INITIAL_RETRY_DELAY

    while retry_count < MAX_RETRIES:
        try:
            async with semaphore:
                # 1) Get the presigned URL with timeout
                try:
                    put_url = await asyncio.wait_for(
                        artifact_manager.put_file(artifact_alias, file_path=relative_path),
                        timeout=CONNECTION_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    print(f"Timeout getting presigned URL for {relative_path}")
                    retry_count += 1
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
                    continue

                # 2) Use aiohttp session to PUT the data with timeout
                try:
                    with open(local_file, "rb") as file_data:
                        async with asyncio.timeout(UPLOAD_TIMEOUT):
                            async with session.put(put_url, data=file_data) as resp:
                                if resp.status != 200:
                                    raise RuntimeError(
                                        f"File upload failed for {local_file}, status={resp.status}"
                                    )
                except asyncio.TimeoutError:
                    print(f"Upload timed out after {UPLOAD_TIMEOUT} seconds for {relative_path}")
                    return False

                # 3) Record successful upload
                upload_record["uploaded_files"].add(relative_path)
                upload_record["completed_files"] += 1

                # 4) Save progress periodically (every 10 files)
                if upload_record["completed_files"] % 10 == 0:
                    save_upload_record(upload_record)

                print(
                    f"Uploaded file: {relative_path} ({upload_record['completed_files']}/{upload_record['total_files']})"
                )
                return True

        except Exception as e:
            print(f"Error uploading {relative_path}: {str(e)}")
            return False

    print(f"Failed to upload {relative_path} after {MAX_RETRIES} attempts")
    return False


async def process_batch(batch, artifact_manager, semaphore, session, upload_record):
    """Process a batch of files with a timeout wrapper, retrying the entire batch if any file fails."""
    while True:  # Keep retrying the batch until all files are successfully uploaded
        tasks = []
        for local_file, relative_path in batch:
            if relative_path in upload_record["uploaded_files"]:
                continue

            task = asyncio.create_task(
                upload_single_file(
                    artifact_manager,
                    ARTIFACT_ALIAS,
                    local_file,
                    relative_path,
                    semaphore,
                    session,
                    upload_record,
                )
            )
            tasks.append((task, local_file, relative_path))

        # Wait for all tasks to complete with a timeout, collect failures
        failed_uploads = []
        for task, local_file, relative_path in tasks:
            try:
                # Set a reasonable timeout for the entire operation
                success = await asyncio.wait_for(task, timeout=UPLOAD_TIMEOUT * 2)
                if not success:
                    failed_uploads.append((local_file, relative_path))
            except asyncio.TimeoutError:
                print(f"Task timed out for {relative_path}")
                # Cancel the task to prevent it from continuing in the background
                if not task.done():
                    task.cancel()
                failed_uploads.append((local_file, relative_path))
            except Exception as e:
                print(f"Exception in task for {relative_path}: {str(e)}")
                failed_uploads.append((local_file, relative_path))

        if not failed_uploads:
            # If there are no failed uploads, break the loop and return
            return []

        print(f"Retrying {len(failed_uploads)} failed uploads in the batch...")
        # Retry the failed uploads in the next iteration of the loop
        batch = failed_uploads


async def retry_upload_with_new_connections(
    local_file, 
    relative_path, 
    upload_record, 
    retries=MAX_RETRIES, 
    initial_delay=INITIAL_RETRY_DELAY
):
    """
    Retry uploading a file with new connections for each attempt.
    This avoids issues with WebSocket connections getting stuck.
    """
    retry_count = 0
    retry_delay = initial_delay
    
    while retry_count < retries:
        if retry_count > 0:
            print(f"Retry {retry_count}/{retries} for {relative_path}")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)  # Exponential backoff
        
        api = None
        try:
            api, artifact_manager = await get_artifact_manager()
            semaphore = asyncio.Semaphore(1)  # Use 1 for individual retries
            
            async with aiohttp.ClientSession() as session:
                task = asyncio.create_task(
                    upload_single_file(
                        artifact_manager,
                        ARTIFACT_ALIAS,
                        local_file,
                        relative_path,
                        semaphore,
                        session,
                        upload_record
                    )
                )
                success = await asyncio.wait_for(task, timeout=UPLOAD_TIMEOUT * 2)
                
                if success:
                    return True
        
        except asyncio.TimeoutError:
            print(f"Connection timed out during retry {retry_count} for {relative_path}")
        except Exception as e:
            print(f"Connection error during retry {retry_count} for {relative_path}: {str(e)}")
        finally:
            if api:
                try:
                    await asyncio.wait_for(api.disconnect(), timeout=5)
                except Exception as e:
                    print(f"Failed to cleanly disconnect API connection for {relative_path}: {e}")
            
        retry_count += 1
    
    print(f"Failed to upload {relative_path} after {retries} attempts")
    return False


async def upload_zarr_file(zarr_file: str) -> bool:
    """Upload a zarr file to the artifact manager."""
    zarr_path = os.path.join(STITCHED_DIR, zarr_file)
    if not os.path.exists(zarr_path):
        print(f"Zarr file {zarr_path} does not exist")
        return False
    
    # Load upload record
    upload_record = load_upload_record()
    if isinstance(upload_record["uploaded_files"], list):
        upload_record["uploaded_files"] = set(upload_record["uploaded_files"])
    
    # Initialize per-folder tracking
    folder_total_files = 0
    folder_completed_files = 0
    
    # Put the dataset in staging mode
    try:
        api, artifact_manager = await get_artifact_manager()
        
        try:
            # Read the current manifest
            dataset_manifest = {
            "name": "image-map-20250410-treatment",
            "description": "The Image Map of U2OS FUCCI Drug Treatment",
            }
            # Put the dataset in staging mode
            print(f"Putting dataset {ARTIFACT_ALIAS} in staging mode...")
            await artifact_manager.edit(
                artifact_id=ARTIFACT_ALIAS,
                manifest=dataset_manifest,  # Preserve the same manifest
                version="stage"    # Put in staging mode
            )
            print("Dataset is now in staging mode")
            
        except Exception as e:
            print(f"Error putting dataset in staging mode: {e}")
            if "not found" in str(e).lower():
                print("Dataset not found. It may need to be created first.")
                # disconnect the connection
                await api.disconnect()
                return False
        finally:
            # disconnect the connection
            try:
                await api.disconnect()
            except:
                pass
    except Exception as e:
        print(f"Failed to setup staging mode: {e}")
        return False
    
    # 1) Prepare a list of (local_file, relative_path) to upload
    to_upload = []
    
    # Extract the folder name without .zarr extension to use as the top-level directory
    folder_name = zarr_file.replace('.zarr', '')
    
    # Walk through the zarr directory
    for root_dir, _, files in os.walk(zarr_path):
        # Skip metadata files at the root level, we only want channel directories
        rel_dir = os.path.relpath(root_dir, zarr_path)
        if rel_dir == '.':
            continue
            
        # Collect files
        for filename in files:
            local_file = os.path.join(root_dir, filename)
            
            # Create a new relative path with folder_name as the top directory
            # and maintaining the internal structure (channel/scale/chunks)
            internal_path = os.path.relpath(root_dir, zarr_path)
            relative_path = os.path.join(folder_name, internal_path, filename)
            
            to_upload.append((local_file, relative_path))
            folder_total_files += 1
    
    # Update total files count while preserving existing completed files
    upload_record["total_files"] = len(to_upload) + upload_record.get("total_files", 0)
    save_upload_record(upload_record)
    
    print(f"\nStarting upload for zarr file: {zarr_file}")
    print(f"Files to upload in this zarr: {folder_total_files}")
    
    # Connect to Artifact Manager
    api, artifact_manager = await get_artifact_manager()
    
    # 2) First attempt to upload files in parallel, but in smaller batches
    BATCH_SIZE = 10  # Process files in smaller batches to limit potential timeouts
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    batches = [to_upload[i:i + BATCH_SIZE] for i in range(0, len(to_upload), BATCH_SIZE)]
    failed_uploads = []
    
    # Process each batch with a fresh session
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} files)")
        try:
            # Use a fresh session for each batch
            async with aiohttp.ClientSession() as session:
                batch_failures = await process_batch(batch, artifact_manager, semaphore, session, upload_record)
                failed_uploads.extend(batch_failures)
                
            # Save progress after each batch
            save_upload_record(upload_record)
            
        except Exception as e:
            print(f"Error processing batch {i+1}: {str(e)}")
            # If a batch fails entirely, add all its files to the failed list
            for local_file, relative_path in batch:
                if relative_path not in upload_record["uploaded_files"]:
                    failed_uploads.append((local_file, relative_path))
    
    # disconnect the initial connection
    try:
        await asyncio.wait_for(api.disconnect(), timeout=10)
    except:
        print("Failed to cleanly disconnect API connection")
    
    # 3) Retry failed uploads one by one with fresh connections
    print(f"First pass completed. Retrying {len(failed_uploads)} failed uploads...")
    
    # Process retries in smaller groups to avoid overloading the connection pool
    retry_batch_size = 5
    for i in range(0, len(failed_uploads), retry_batch_size):
        retry_batch = failed_uploads[i:i + retry_batch_size]
        retry_tasks = []
        
        for local_file, relative_path in retry_batch:
            task = asyncio.create_task(
                retry_upload_with_new_connections(
                    local_file, 
                    relative_path, 
                    upload_record
                )
            )
            retry_tasks.append(task)
        
        # Wait for this batch of retries to complete
        await asyncio.gather(*retry_tasks)
        
        # Save progress after each retry batch
        save_upload_record(upload_record)
    
    # 4) Save final record and get a fresh connection to commit
    save_upload_record(upload_record)
    
    # Final connection for commit with retries
    commit_success = False
    commit_attempts = 0
    
    while not commit_success and commit_attempts < 5:
        try:
            api, artifact_manager = await get_artifact_manager()
            await asyncio.wait_for(
                artifact_manager.commit(ARTIFACT_ALIAS),
                timeout=CONNECTION_TIMEOUT
            )
            print("Dataset committed successfully.")
            commit_success = True
        except Exception as e:
            commit_attempts += 1
            print(f"Error committing dataset (attempt {commit_attempts}/5): {str(e)}")
            await asyncio.sleep(5)
        finally:
            if api:
                try:
                    await asyncio.wait_for(api.disconnect(), timeout=5)
                except:
                    pass
    
    if not commit_success:
        print("WARNING: Failed to commit the dataset after multiple attempts.")
        return False
    
    print(
        f"Zarr upload complete: {folder_completed_files}/{folder_total_files} files uploaded in {zarr_file}\n"
        f"Total files uploaded across all zarr files: {upload_record['completed_files']}/{upload_record['total_files']}"
    )
    return True


async def process_folder(folder_name: str) -> bool:
    """Process a single folder through stitching, uploading, and cleanup."""
    folder_path = os.path.join(BASE_DIR, folder_name)
    
    # Step 1: Check for .done file
    folder_datetime = extract_datetime_from_folder(folder_name)
    zarr_filename = f"{folder_datetime}.zarr" if folder_datetime else f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zarr"
    done_file_path = os.path.join(STITCHED_DIR, f"{zarr_filename}.done")
    
    if os.path.exists(done_file_path):
        print(f".done file exists for {zarr_filename}, skipping stitching.")
    else:
        # Step 2: Stitch the folder
        print(f"\nStarting stitching process for folder: {folder_name}")
        stitch_success = stitch_folder(folder_path)
        if not stitch_success:
            print(f"Failed to stitch folder: {folder_name}")
            return False
        
        # Create .done file after successful stitching
        with open(done_file_path, 'w') as f:
            f.write("Stitching completed.")
    
    # Step 3: Get the stitched zarr file
    zarr_files = get_zarr_files()
    if not zarr_files:
        print(f"No zarr files were created from stitching folder: {folder_name}")
        return False
    
    # Find the matching zarr file for this folder
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