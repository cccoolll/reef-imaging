import os
import re
import asyncio
import time
import importlib.util
import sys
from datetime import datetime
import glob
import threading
import shutil
import concurrent.futures
from typing import List, Optional, Dict, Set
from queue import Queue

# Import the existing scripts as modules
def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Path to the existing scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_TREATMENT_DATA_PATH = os.path.join(SCRIPT_DIR, "hypha_tools/artifact_manager/upload_treatment_data.py")
STITCH_ZARR_PATH = os.path.join(SCRIPT_DIR, "hypha_tools/artifact_manager/image_processing/stitch_zarr.py")
UPLOAD_ZARR_PATH = os.path.join(SCRIPT_DIR, "hypha_tools/artifact_manager/step1_upload_zarr.py")

# Import modules
upload_treatment_data = import_module_from_file("upload_treatment_data", UPLOAD_TREATMENT_DATA_PATH)
stitch_zarr = import_module_from_file("stitch_zarr", STITCH_ZARR_PATH)
upload_zarr = import_module_from_file("upload_zarr", UPLOAD_ZARR_PATH)

# Constants
DATA_ROOT = "/media/reef/harddisk"
EXPERIMENT_ID = "20250410-fucci-time-lapse-scan"
FOLDER_PATTERN = f"{EXPERIMENT_ID}_*"
CHECK_INTERVAL = 300  # Check for new folders every 5 minutes

# Global state tracking
raw_upload_queue = Queue()
stitch_upload_queue = Queue()
uploaded_raw_folders = set()
processed_stitch_folders = set()
lock = threading.Lock()
stop_threads = False  # Flag to signal threads to stop

# Force parallel processing by making the stitching thread immediately 
# pick up folders for processing, without waiting for the raw upload to complete
parallel_processing = True  # Set to True to enable true parallel processing

def get_experiment_folders() -> List[str]:
    """Get all folders matching the experiment ID pattern."""
    folder_pattern = os.path.join(DATA_ROOT, FOLDER_PATTERN)
    folders = glob.glob(folder_pattern)
    # Sort folders by acquisition time
    folders.sort()
    return folders

def get_folder_timestamp(folder_path: str) -> datetime:
    """Extract timestamp from folder name."""
    # Format: 20250410-fucci-time-lapse-scan_2025-04-10_13-50-7.762411
    match = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{1,2}\.\d+)$', folder_path)
    if match:
        datetime_str = match.group(1).replace('-', ':').replace('_', ' ')
        # Handle decimal seconds separately
        main_part, decimal_part = datetime_str.rsplit('.', 1)
        try:
            dt = datetime.strptime(main_part, '%Y:%m:%d %H:%M:%S')
            return dt
        except ValueError:
            print(f"Error parsing timestamp from {folder_path}")
    return datetime.fromtimestamp(os.path.getctime(folder_path))

def get_processed_folders() -> Dict[str, Set[str]]:
    """Get list of already processed folders from tracking files."""
    result = {
        "raw_uploaded": set(),
        "stitch_processed": set()
    }
    
    # Raw upload record
    raw_upload_file = "raw_upload_record.txt"
    if os.path.exists(raw_upload_file):
        with open(raw_upload_file, "r") as f:
            result["raw_uploaded"] = set(line.strip() for line in f if line.strip())
    
    # Stitch process record
    stitch_process_file = "stitch_process_record.txt"
    if os.path.exists(stitch_process_file):
        with open(stitch_process_file, "r") as f:
            result["stitch_processed"] = set(line.strip() for line in f if line.strip())
    
    return result

def mark_folder_as_processed(folder_path: str, process_type: str):
    """Mark a folder as processed in the tracking file."""
    if process_type == "raw_upload":
        record_file = "raw_upload_record.txt"
    else:
        record_file = "stitch_process_record.txt"
    
    with lock:
        with open(record_file, "a") as f:
            f.write(f"{folder_path}\n")

async def process_raw_upload(folder_path: str):
    """Process raw imaging data upload for a single folder."""
    folder_name = os.path.basename(folder_path)
    print(f"\n{'='*80}\n[RAW UPLOAD] Processing folder: {folder_name}\n{'='*80}")
    
    try:
        # 0) First, put the dataset in staging mode before uploading
        print(f"\n--- Step 0: Putting dataset in staging mode ---")
        try:
            # Get artifact manager
            api, artifact_manager = await upload_treatment_data.get_artifact_manager()
            
            # Put the dataset in staging mode
            # Read the current manifest first to preserve it
            try:
                dataset = await artifact_manager.read(artifact_id=upload_treatment_data.DATASET_ALIAS)
                print(f"Successfully read current dataset manifest for {upload_treatment_data.DATASET_ALIAS}")
            except Exception as e:
                print(f"Error reading dataset manifest, will create new: {str(e)}")
                dataset = {"name": upload_treatment_data.DATASET_ALIAS, "description": "Treatment data"}
            
            # Put the dataset in staging mode
            await artifact_manager.edit(
                artifact_id=upload_treatment_data.DATASET_ALIAS,
                manifest=dataset,  # Preserve the same manifest
                version="stage"    # Put in staging mode
            )
            print(f"Successfully put dataset {upload_treatment_data.DATASET_ALIAS} in staging mode")
            
            # Close the API connection
            await api.close()
            
        except Exception as e:
            print(f"Error putting dataset in staging mode: {str(e)}")
            # If this fails, perhaps the dataset doesn't exist yet, and will be created
            # by the upload_treatment_data.main() function
            
        # 1) Temporarily modify SOURCE_DIRS to only include the current folder
        original_source_dirs = upload_treatment_data.SOURCE_DIRS
        upload_treatment_data.SOURCE_DIRS = [folder_path]
        
        # 2) Run the upload_treatment_data script
        await upload_treatment_data.main()
        
        # 3) Restore original SOURCE_DIRS
        upload_treatment_data.SOURCE_DIRS = original_source_dirs
        
        # 4) Mark folder as processed for raw upload
        mark_folder_as_processed(folder_path, "raw_upload")
        
        if not parallel_processing:
            # 5) Add to the stitch queue if not using parallel processing
            # When using parallel processing, the stitch thread checks for completed
            # raw uploads independently
            with lock:
                stitch_upload_queue.put(folder_path)
        
        print(f"\n[RAW UPLOAD] Completed for {folder_name}!")
        return True
    except Exception as e:
        print(f"Error uploading raw data for {folder_name}: {str(e)}")
        return False

async def process_stitch_upload(folder_path: str):
    """Process stitching and zarr upload for a single folder."""
    folder_name = os.path.basename(folder_path)
    print(f"\n{'='*80}\n[STITCH WORKER] Processing folder: {folder_name}\n{'='*80}")
    
    try:
        # Step 1: Stitch the tiles using stitch_zarr
        print(f"\n--- Step 1: Stitching tiles for {folder_name} ---")
        
        # Run stitching for this folder
        stitch_zarr_single_folder(folder_path)
        
        # Step 2: Preparing for zarr upload
        print(f"\n--- Step 2: Putting image-map dataset in staging mode ---")
        # Extract date and time from the folder name
        date_time_str = folder_name.split('_')[1] + '_' + folder_name.split('_')[2].split('.')[0]
        zarr_file = f"/media/reef/harddisk/test_stitch_zarr/{date_time_str}.zarr"
        
        # First, put the zarr artifact in staging mode
        try:
            # Get artifact manager
            api, artifact_manager = await upload_zarr.get_artifact_manager()
            
            # Put the dataset in staging mode
            try:
                dataset = await artifact_manager.read(artifact_id=upload_zarr.ARTIFACT_ALIAS)
                print(f"Successfully read current zarr dataset manifest for {upload_zarr.ARTIFACT_ALIAS}")
            except Exception as e:
                print(f"Error reading zarr dataset manifest, will create new: {str(e)}")
                dataset = {"name": upload_zarr.ARTIFACT_ALIAS, "description": "Image map data"}
            
            # Put the dataset in staging mode
            await artifact_manager.edit(
                artifact_id=upload_zarr.ARTIFACT_ALIAS,
                manifest=dataset,  # Preserve the same manifest
                version="stage"    # Put in staging mode
            )
            print(f"Successfully put zarr dataset {upload_zarr.ARTIFACT_ALIAS} in staging mode")
            
            # Close the API connection
            await api.close()
            
        except Exception as e:
            print(f"Error putting zarr dataset in staging mode: {str(e)}")
        
        # Step 3: Upload stitched zarr data
        print(f"\n--- Step 3: Uploading stitched zarr data for {folder_name} ---")
        
        # Temporarily modify ORIGINAL_ZARR_PATHS to only include the current zarr file
        original_zarr_paths = upload_zarr.ORIGINAL_ZARR_PATHS
        upload_zarr.ORIGINAL_ZARR_PATHS = [zarr_file]
        
        # Run the upload_zarr script
        await upload_zarr.main()
        
        # Restore original ORIGINAL_ZARR_PATHS
        upload_zarr.ORIGINAL_ZARR_PATHS = original_zarr_paths
        
        # Step 4: Remove zarr file to save disk space
        print(f"\n--- Step 4: Removing zarr file to save disk space ---")
        if os.path.exists(zarr_file) and os.path.isdir(zarr_file):
            try:
                shutil.rmtree(zarr_file)
                print(f"Successfully removed zarr file: {zarr_file}")
            except Exception as e:
                print(f"Warning: Failed to remove zarr file {zarr_file}: {str(e)}")
        
        # Mark folder as processed for stitching
        mark_folder_as_processed(folder_path, "stitch_process")
        
        print(f"\n[STITCH WORKER] Completed for {folder_name}!")
        return True
    except Exception as e:
        print(f"Error in stitching/zarr upload for {folder_name}: {str(e)}")
        return False

def stitch_zarr_single_folder(folder_path):
    """Run stitch_zarr for a single folder."""
    # Extract date and time from the folder name
    folder_name = os.path.basename(folder_path)
    date_time_str = folder_name.split('_')[1] + '_' + folder_name.split('_')[2].split('.')[0]
    zarr_filename = f"{date_time_str}.zarr"
    
    # Setup parameters for this specific folder
    image_folder = os.path.join(folder_path, "0")
    parameter_file = os.path.join(folder_path, "acquisition parameters.json")
    coordinates_file = os.path.join(image_folder, "coordinates.csv")
    output_folder = "/media/reef/harddisk/test_stitch_zarr"
    os.makedirs(output_folder, exist_ok=True)

    # Load imaging parameters and coordinates
    parameters = stitch_zarr.load_imaging_parameters(parameter_file)
    
    # Check if coordinates file exists, if not generate it
    if not os.path.exists(coordinates_file):
        print("Coordinates file not found, generating...")
        config = {
            "dx(mm)": 0.9,
            "Nx": 1,
            "dy(mm)": 0.9,
            "Ny": 1,
            "dz(um)": 1.5,
            "Nz": 1,
            "dt(s)": 0.0,
            "Nt": 1,
            "with AF": True,
            "with reflection AF": False,
            "objective": {"magnification": 20, "NA": 0.4, "tube_lens_f_mm": 180, "name": "20x"},
            "sensor_pixel_size_um": 1.85,
            "tube_lens_mm": 50
        }
        class WELLPLATE_FORMAT_96:
            NUMBER_OF_SKIP = 0
            WELL_SIZE_MM = 6.21
            WELL_SPACING_MM = 9
            A1_X_MM = 14.3
            A1_Y_MM = 11.36
        
        coordinates = stitch_zarr.generate_coordinates_file(image_folder, config, WELLPLATE_FORMAT_96)
    else:
        coordinates = stitch_zarr.pd.read_csv(coordinates_file)

    # Define stage limits
    stage_limits = {
        "x_positive": 120,
        "x_negative": 0,
        "y_positive": 86,
        "y_negative": 0,
        "z_positive": 6
    }
    
    # Get pixel size and calculate canvas size
    pixel_size_xy = stitch_zarr.get_pixel_size(parameters)
    canvas_width, canvas_height = stitch_zarr.create_canvas_size(stage_limits, pixel_size_xy)

    # Calculate appropriate number of pyramid levels
    max_dimension = max(canvas_width, canvas_height)
    max_levels = int(stitch_zarr.np.floor(stitch_zarr.np.log2(max_dimension)) // 2)
    pyramid_levels = 6
    
    # Parse image filenames and get unique channels
    image_info = stitch_zarr.parse_image_filenames(image_folder)
    channels = list(set(info["channel_name"] for info in image_info))
    selected_channel = ['Fluorescence_561_nm_Ex', 'Fluorescence_488_nm_Ex', 'BF_LED_matrix_full']
    
    # Check if dataset exists, if so, use it
    zarr_file_path = os.path.join(output_folder, zarr_filename)
    if os.path.exists(zarr_file_path):
        print(f"Dataset {zarr_filename} exists, opening in append mode.")
        datasets = stitch_zarr.zarr.open(zarr_file_path, mode="a")
    else:
        # Create new OME-NGFF file
        root, datasets = stitch_zarr.create_ome_ngff(output_folder, canvas_width, canvas_height, 
                                                    selected_channel, zarr_filename=zarr_filename, 
                                                    pyramid_levels=pyramid_levels)
    
    # Process images and stitch them
    stitch_zarr.process_images(image_info, coordinates, datasets, pixel_size_xy, stage_limits, 
                             selected_channel=selected_channel, pyramid_levels=pyramid_levels)
    print(f"Stitching completed for {folder_path}")

# Thread function for raw data uploads
def raw_upload_thread_func():
    """Thread function for processing raw upload queue."""
    print("Starting raw upload worker thread")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    while not stop_threads:
        if raw_upload_queue.empty():
            time.sleep(5)  # Shorter wait time to be more responsive
            continue
        
        folder_path = raw_upload_queue.get()
        try:
            # Run the async function in this thread's event loop
            print(f"[RAW UPLOAD] Starting processing of {os.path.basename(folder_path)}")
            loop.run_until_complete(process_raw_upload(folder_path))
            
            # If parallel processing is enabled, the stitching thread will pick up
            # folders independently. Otherwise, we add to the stitch queue here.
            if not parallel_processing:
                with lock:
                    if folder_path not in processed_stitch_folders:
                        print(f"Adding {os.path.basename(folder_path)} to stitch queue")
                        stitch_upload_queue.put(folder_path)
                        
        except Exception as e:
            print(f"Error in raw upload thread: {str(e)}")
        finally:
            raw_upload_queue.task_done()
    
    loop.close()
    print("Raw upload thread stopped")

# Thread function for stitching and zarr uploads
def stitch_upload_thread_func():
    """Thread function for processing stitch and zarr upload queue."""
    print("Starting stitch and zarr upload worker thread")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # When using parallel processing, we check for already processed raw folders
    # and add them to our queue immediately, rather than waiting for the raw thread
    if parallel_processing:
        processed_data = loop.run_until_complete(get_processed_folders_async())
        with lock:
            for folder_path in processed_data["raw_uploaded"]:
                if folder_path not in processed_stitch_folders and folder_path not in list(stitch_upload_queue.queue):
                    print(f"[STITCH WORKER] Adding {os.path.basename(folder_path)} to stitch queue from previous raw uploads")
                    stitch_upload_queue.put(folder_path)
    
    while not stop_threads:
        # In parallel mode, we also directly check for newly completed raw uploads
        if parallel_processing and stitch_upload_queue.empty():
            try:
                processed_data = loop.run_until_complete(get_processed_folders_async())
                with lock:
                    for folder_path in processed_data["raw_uploaded"]:
                        if folder_path not in processed_stitch_folders and folder_path not in list(stitch_upload_queue.queue):
                            print(f"[STITCH WORKER] Adding {os.path.basename(folder_path)} to stitch queue from raw uploads")
                            stitch_upload_queue.put(folder_path)
            except Exception as e:
                print(f"Error checking for new raw uploads: {str(e)}")
                
        if stitch_upload_queue.empty():
            time.sleep(5)  # Shorter wait time
            continue
        
        folder_path = stitch_upload_queue.get()
        try:
            # Run the async function in this thread's event loop
            print(f"[STITCH WORKER] Starting processing of {os.path.basename(folder_path)}")
            loop.run_until_complete(process_stitch_upload(folder_path))
        except Exception as e:
            print(f"Error in stitch upload thread: {str(e)}")
        finally:
            stitch_upload_queue.task_done()
    
    loop.close()
    print("Stitch upload thread stopped")

# Monitor thread function
def monitor_thread_func():
    """Thread function for monitoring for new folders."""
    print("Starting folder monitor thread")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Initialize processed folder records
    processed_data = loop.run_until_complete(get_processed_folders_async())
    with lock:
        uploaded_raw_folders.update(processed_data["raw_uploaded"])
        processed_stitch_folders.update(processed_data["stitch_processed"])
    
    while not stop_threads:
        try:
            loop.run_until_complete(monitor_folders_once())
            time.sleep(CHECK_INTERVAL)
        except Exception as e:
            print(f"Error in monitor thread: {str(e)}")
            time.sleep(60)  # Shorter wait on error
    
    loop.close()
    print("Monitor thread stopped")

async def get_processed_folders_async() -> Dict[str, Set[str]]:
    """Async version of get_processed_folders."""
    result = {
        "raw_uploaded": set(),
        "stitch_processed": set()
    }
    
    # Raw upload record
    raw_upload_file = "raw_upload_record.txt"
    if os.path.exists(raw_upload_file):
        with open(raw_upload_file, "r") as f:
            result["raw_uploaded"] = set(line.strip() for line in f if line.strip())
    
    # Stitch process record
    stitch_process_file = "stitch_process_record.txt"
    if os.path.exists(stitch_process_file):
        with open(stitch_process_file, "r") as f:
            result["stitch_processed"] = set(line.strip() for line in f if line.strip())
    
    return result

async def monitor_folders_once():
    """Monitor for new folders and add them to the processing queues (single iteration)."""
    # Get all current folders for the experiment
    all_folders = get_experiment_folders()
    
    if not all_folders:
        print(f"No folders found matching pattern {FOLDER_PATTERN}")
        return
    
    # Skip the latest folder which may still be acquiring data
    folders_to_process = all_folders[:-1]
    
    # Filter out already processed folders for raw upload
    with lock:
        raw_unprocessed = [f for f in folders_to_process if f not in uploaded_raw_folders]
    
    if raw_unprocessed:
        print(f"Found {len(raw_unprocessed)} folder(s) for raw upload")
        for folder_path in raw_unprocessed:
            with lock:
                if folder_path not in uploaded_raw_folders:
                    print(f"Adding {os.path.basename(folder_path)} to raw upload queue")
                    raw_upload_queue.put(folder_path)
                    uploaded_raw_folders.add(folder_path)
    
    # For completed raw uploads that haven't been stitched yet
    processed_data = await get_processed_folders_async()
    for folder_path in processed_data["raw_uploaded"]:
        with lock:
            if folder_path not in processed_stitch_folders and folder_path not in list(stitch_upload_queue.queue):
                print(f"Adding {os.path.basename(folder_path)} to stitch queue from previous raw uploads")
                stitch_upload_queue.put(folder_path)
    
    # Report queue status
    print(f"Queues: Raw upload ({raw_upload_queue.qsize()}), Stitch upload ({stitch_upload_queue.qsize()})")
    print(f"Next check in {CHECK_INTERVAL//60} minutes...")

def start_worker_threads():
    """Start the worker threads for parallel processing."""
    global stop_threads
    stop_threads = False
    
    # Create threads
    raw_thread = threading.Thread(target=raw_upload_thread_func, daemon=True)
    stitch_thread = threading.Thread(target=stitch_upload_thread_func, daemon=True)
    monitor_thread = threading.Thread(target=monitor_thread_func, daemon=True)
    
    # Start threads
    raw_thread.start()
    stitch_thread.start()
    monitor_thread.start()
    
    return [raw_thread, stitch_thread, monitor_thread]

def stop_worker_threads(threads):
    """Stop the worker threads."""
    global stop_threads
    stop_threads = True
    
    # Wait for threads to finish
    for thread in threads:
        if thread.is_alive():
            thread.join(timeout=5)
    
    print("All worker threads stopped")

async def main():
    """Main entry point for the script."""
    print(f"Automated Imaging Workflow")
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Parallel processing: {'Enabled' if parallel_processing else 'Disabled'}")
    
    # Get all experiment folders
    all_folders = get_experiment_folders()
    if not all_folders:
        print(f"No folders found matching pattern {FOLDER_PATTERN}")
        return
    
    # Display folders to the user
    print("\nAvailable folders:")
    for i, folder in enumerate(all_folders):
        folder_name = os.path.basename(folder)
        print(f"{i+1}. {folder_name}")
    
    # Ask user which folder to start with
    while True:
        try:
            choice = input("\nEnter the folder name or number to start with: ")
            
            # Check if user input is a number
            if choice.isdigit() and 1 <= int(choice) <= len(all_folders):
                selected_folder = all_folders[int(choice)-1]
                break
            
            # Check if user input matches a folder
            matching_folders = [f for f in all_folders if os.path.basename(f) == choice]
            if matching_folders:
                selected_folder = matching_folders[0]
                break
                
            print("Invalid choice. Please enter a valid folder name or number.")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Get the index of the selected folder
    start_index = all_folders.index(selected_folder)
    
    # Process the selected folder and any subsequent folders (except the latest one)
    processed_data = await get_processed_folders_async()
    
    # Initialize queues for folders to process
    for folder_path in all_folders[start_index:-1]:
        if folder_path not in processed_data["raw_uploaded"]:
            print(f"Adding {os.path.basename(folder_path)} to raw upload queue")
            raw_upload_queue.put(folder_path)
            
        # In parallel mode, we also add to stitch queue if raw is already done
        if parallel_processing and folder_path in processed_data["raw_uploaded"] and folder_path not in processed_data["stitch_processed"]:
            print(f"Adding {os.path.basename(folder_path)} to stitch upload queue (parallel mode)")
            stitch_upload_queue.put(folder_path)
    
    # Start worker threads for parallel processing
    print("\nStarting worker threads for true parallel processing...\n")
    threads = start_worker_threads()
    
    try:
        # Keep the main thread running to handle user input
        while True:
            # Check if all threads are still alive
            if not all(thread.is_alive() for thread in threads):
                print("One of the worker threads stopped unexpectedly, restarting...")
                stop_worker_threads(threads)
                threads = start_worker_threads()
            
            # Status report
            print(f"\nStatus: Raw upload queue: {raw_upload_queue.qsize()}, Stitch queue: {stitch_upload_queue.qsize()}")
            print(f"Raw processed: {len(processed_data['raw_uploaded'])}, Stitch processed: {len(processed_data['stitch_processed'])}")
            
            # Sleep for a bit to avoid busy waiting
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, stopping worker threads...")
        stop_worker_threads(threads)
        print("Exiting.")

if __name__ == "__main__":
    asyncio.run(main()) 