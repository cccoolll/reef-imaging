import os
import sys
import time
import asyncio
import shutil
from datetime import datetime
from typing import List, Optional
import re

# Import the existing scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import image_processing.stitch_zarr as stitch_zarr
import step1_upload_zarr

# Constants
BASE_DIR = "/media/reef/harddisk"
EXPERIMENT_ID = "20250410-fucci-time-lapse-scan"
STITCHED_DIR = "/media/reef/harddisk/test_stitch_zarr"
STITCH_RECORD_FILE = "stitch_upload_progress.txt"
CHECK_INTERVAL = 60  # Check for new folders every 60 seconds

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
                if len(parts) == 2:
                    processed.append(parts[1])
    
    return processed

def save_processed_folder(folder_name: str, status: str = "completed"):
    """Save the processed folder to the record file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(STITCH_RECORD_FILE, "a") as f:
        f.write(f"{timestamp} - {folder_name} - {status}\n")

def stitch_folder(folder_name: str) -> bool:
    """Process a folder using stitch_zarr.py."""
    full_path = os.path.join(BASE_DIR, folder_name)
    if not os.path.exists(full_path):
        print(f"Folder {full_path} does not exist!")
        return False
    
    print(f"Stitching folder: {folder_name}")
    
    # Save original data_folders
    original_data_folders = stitch_zarr.data_folders if hasattr(stitch_zarr, 'data_folders') else []
    
    try:
        # Update paths in stitch_zarr
        stitch_zarr.data_folders = [full_path]
        
        # Run the stitching process
        stitch_zarr.main()
        print(f"Successfully stitched folder: {folder_name}")
        return True
    except Exception as e:
        print(f"Error stitching folder {folder_name}: {e}")
        return False
    finally:
        # Restore original data_folders
        if hasattr(stitch_zarr, 'data_folders'):
            stitch_zarr.data_folders = original_data_folders

async def upload_zarr_files() -> bool:
    """Upload all zarr files using step1_upload_zarr.py."""
    print("Uploading zarr files...")
    
    # Get zarr file paths
    zarr_files = get_zarr_files()
    if not zarr_files:
        print("No zarr files found to upload")
        return False
    
    # Save original zarr paths
    original_zarr_paths = step1_upload_zarr.ORIGINAL_ZARR_PATHS.copy()
    
    try:
        # First, put the dataset in staging mode
        api, artifact_manager = await step1_upload_zarr.get_artifact_manager()
        
        try:
            # Read the current manifest
            dataset = await artifact_manager.read(
                artifact_id=step1_upload_zarr.ARTIFACT_ALIAS,
                silent=True
            )
            
            # Put the dataset in staging mode
            print(f"Putting dataset {step1_upload_zarr.ARTIFACT_ALIAS} in staging mode...")
            await artifact_manager.edit(
                artifact_id=step1_upload_zarr.ARTIFACT_ALIAS,
                manifest=dataset,  # Preserve the same manifest
                version="stage"    # Put in staging mode
            )
            print("Dataset is now in staging mode")
            
        except Exception as e:
            print(f"Error putting dataset in staging mode: {e}")
            if "not found" in str(e).lower():
                print("Dataset not found. It may need to be created first.")
                # Close the connection
                await api.close()
                return False
        finally:
            # Close the connection
            try:
                await api.close()
            except:
                pass
        
        # Update paths in upload script
        zarr_paths = [os.path.join(STITCHED_DIR, zarr_file) for zarr_file in zarr_files]
        step1_upload_zarr.ORIGINAL_ZARR_PATHS = zarr_paths
        
        # Run the upload process
        await step1_upload_zarr.main()
        print("Successfully uploaded zarr files")
        return True
    except Exception as e:
        print(f"Error uploading zarr files: {e}")
        return False
    finally:
        # Restore original zarr paths
        step1_upload_zarr.ORIGINAL_ZARR_PATHS = original_zarr_paths

def get_zarr_files() -> List[str]:
    """Get list of zarr files in the stitched directory."""
    if not os.path.exists(STITCHED_DIR):
        return []
    
    return [f for f in os.listdir(STITCHED_DIR) if f.endswith('.zarr')]

def cleanup_zarr_files():
    """Delete zarr files after successful upload."""
    zarr_files = get_zarr_files()
    
    for zarr_file in zarr_files:
        zarr_path = os.path.join(STITCHED_DIR, zarr_file)
        try:
            print(f"Removing zarr file: {zarr_path}")
            if os.path.isdir(zarr_path):
                shutil.rmtree(zarr_path)
            else:
                os.remove(zarr_path)
        except Exception as e:
            print(f"Error removing {zarr_path}: {e}")

async def stitch_and_upload_folder(folder_name: str) -> bool:
    """Process a single folder - stitch, upload, and cleanup."""
    full_path = os.path.join(BASE_DIR, folder_name)
    if not os.path.exists(full_path):
        print(f"Folder {full_path} does not exist!")
        return False
    
    print(f"Processing folder: {folder_name}")
    
    # Step 1: Stitch the folder
    stitch_success = stitch_folder(folder_name)
    if not stitch_success:
        print(f"Failed to stitch {folder_name}")
        return False
    
    # Step 2: Get the zarr file created for this folder
    zarr_files = get_zarr_files()
    if not zarr_files:
        print("No zarr files were created from stitching")
        return False
    
    # Get the specific zarr file for this folder
    folder_datetime = extract_datetime_from_folder(folder_name)
    if folder_datetime:
        matching_zarr = [f for f in zarr_files if folder_datetime in f]
        if matching_zarr:
            specific_zarr = matching_zarr[0]
            print(f"Found matching zarr file: {specific_zarr}")
            
            # Step 3: Upload the specific zarr file
            upload_success = await upload_single_zarr(specific_zarr)
            
            if upload_success:
                # Step 4: Delete the zarr file if upload was successful
                cleanup_single_zarr(specific_zarr)
                return True
    
    print("Failed to match zarr file with the folder or upload failed")
    return False

def extract_datetime_from_folder(folder_name: str) -> Optional[str]:
    """Extract the datetime string from a folder name."""
    pattern = r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d+)"
    match = re.search(pattern, folder_name)
    if match:
        return match.group(1)
    return None

async def upload_single_zarr(zarr_file: str) -> bool:
    """Upload a single zarr file."""
    print(f"Uploading zarr file: {zarr_file}")
    
    # Save original zarr paths
    original_zarr_paths = step1_upload_zarr.ORIGINAL_ZARR_PATHS.copy()
    
    try:
        # First, put the dataset in staging mode
        api, artifact_manager = await step1_upload_zarr.get_artifact_manager()
        
        try:
            # Read the current manifest
            dataset = await artifact_manager.read(
                artifact_id=step1_upload_zarr.ARTIFACT_ALIAS,
                silent=True
            )
            
            # Put the dataset in staging mode
            print(f"Putting dataset {step1_upload_zarr.ARTIFACT_ALIAS} in staging mode...")
            await artifact_manager.edit(
                artifact_id=step1_upload_zarr.ARTIFACT_ALIAS,
                manifest=dataset,  # Preserve the same manifest
                version="stage"    # Put in staging mode
            )
            print("Dataset is now in staging mode")
            
        except Exception as e:
            print(f"Error putting dataset in staging mode: {e}")
            if "not found" in str(e).lower():
                print("Dataset not found. It may need to be created first.")
                # Close the connection
                await api.close()
                return False
        finally:
            # Close the connection
            try:
                await api.close()
            except:
                pass
        
        # Update paths in upload script with just this single zarr file
        zarr_path = os.path.join(STITCHED_DIR, zarr_file)
        step1_upload_zarr.ORIGINAL_ZARR_PATHS = [zarr_path]
        
        # Run the upload process
        await step1_upload_zarr.main()
        print(f"Successfully uploaded zarr file: {zarr_file}")
        return True
    except Exception as e:
        print(f"Error uploading zarr file {zarr_file}: {e}")
        return False
    finally:
        # Restore original zarr paths
        step1_upload_zarr.ORIGINAL_ZARR_PATHS = original_zarr_paths

def cleanup_single_zarr(zarr_file: str):
    """Delete a single zarr file after successful upload."""
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
    
    start_idx = all_folders.index(start_folder)
    print(f"Starting with folder: {start_folder}")
    
    # Process folders continuously, starting from the selected folder
    current_idx = start_idx
    
    while True:
        all_folders = get_timelapse_folders()  # Refresh the folder list
        
        if current_idx >= len(all_folders) - 1:
            # Wait for new folders to appear
            print(f"Waiting for new folders. Currently at: {all_folders[-1] if all_folders else 'None'}")
            time.sleep(CHECK_INTERVAL)
            continue
        
        # Process the current folder
        folder_to_process = all_folders[current_idx]
        
        # Skip if already processed
        if folder_to_process in processed_folders:
            print(f"Folder {folder_to_process} already processed, skipping.")
            current_idx += 1
            continue
        
        # Process this folder - stitch, upload, delete
        success = await stitch_and_upload_folder(folder_to_process)
        
        if success:
            # Record success and move to next folder
            save_processed_folder(folder_to_process)
            processed_folders.append(folder_to_process)
            current_idx += 1
            print(f"Completed processing folder: {folder_to_process}")
        else:
            # If failed, wait and retry the same folder
            print(f"Failed to process {folder_to_process}. Retrying in {CHECK_INTERVAL} seconds...")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    # Ensure stitched directory exists
    os.makedirs(STITCHED_DIR, exist_ok=True)
    asyncio.run(main()) 