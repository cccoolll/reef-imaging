import os
import sys
import time
import asyncio
from datetime import datetime
import re
from typing import List, Optional

# Import the existing upload script
import upload_treatment_data

# Constants
BASE_DIR = "/media/reef/harddisk"
EXPERIMENT_ID = "20250410-fucci-time-lapse-scan"
UPLOAD_RECORD_FILE = "treatment_upload_progress.txt"
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

def get_last_processed_folder() -> Optional[str]:
    """Get the last processed folder from the record file."""
    if not os.path.exists(UPLOAD_RECORD_FILE):
        return None
    
    with open(UPLOAD_RECORD_FILE, "r") as f:
        lines = f.readlines()
        if lines:
            last_line = lines[-1].strip()
            # The format is: "YYYY-MM-DD HH:MM:SS - folder_name"
            parts = last_line.split(" - ", 1)
            if len(parts) == 2:
                return parts[1]
    
    return None

def save_processed_folder(folder_name: str):
    """Save the processed folder to the record file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(UPLOAD_RECORD_FILE, "a") as f:
        f.write(f"{timestamp} - {folder_name}\n")

async def process_folder(folder_name: str):
    """Process a folder by updating upload_treatment_data settings and running it."""
    full_path = os.path.join(BASE_DIR, folder_name)
    if not os.path.exists(full_path):
        print(f"Folder {full_path} does not exist!")
        return False
    
    print(f"Processing folder: {folder_name}")
    
    # Update SOURCE_DIRS in upload_treatment_data
    upload_treatment_data.SOURCE_DIRS = [full_path]
    
    # Run the upload process
    try:
        # First, put the dataset in staging mode
        api, artifact_manager = await upload_treatment_data.get_artifact_manager()
        
        try:
            # Read the current manifest
            dataset = await artifact_manager.read(
                artifact_id=upload_treatment_data.DATASET_ALIAS,
                silent=True
            )
            
            # Put the dataset in staging mode
            print(f"Putting dataset {upload_treatment_data.DATASET_ALIAS} in staging mode...")
            await artifact_manager.edit(
                artifact_id=upload_treatment_data.DATASET_ALIAS,
                manifest=dataset,  # Preserve the same manifest
                version="stage"    # Put in staging mode
            )
            print("Dataset is now in staging mode")
            
        except Exception as e:
            print(f"Error putting dataset in staging mode: {e}")
            if "not found" in str(e).lower():
                print("Dataset not found. It may need to be created first.")
                return False
            
        finally:
            # Close the connection
            try:
                await api.close()
            except:
                pass
        
        # Now run the main upload process
        await upload_treatment_data.main()
        print(f"Successfully processed folder: {folder_name}")
        return True
    except Exception as e:
        print(f"Error processing folder {folder_name}: {e}")
        return False

async def main():
    # Ask user which folder to start with
    all_folders = get_timelapse_folders()
    
    if not all_folders:
        print(f"No folders matching {EXPERIMENT_ID} found in {BASE_DIR}")
        return
    
    last_processed = get_last_processed_folder()
    
    print("Available folders:")
    for i, folder in enumerate(all_folders):
        status = " (last processed)" if folder == last_processed else ""
        print(f"{i+1}: {folder}{status}")
    
    start_idx = 0
    if last_processed:
        try:
            start_idx = all_folders.index(last_processed) + 1
        except ValueError:
            start_idx = 0
    
    if start_idx >= len(all_folders):
        start_idx = len(all_folders) - 2  # Default to second-to-last folder
    
    default_folder = all_folders[start_idx] if start_idx < len(all_folders) else all_folders[-2]
    
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
        success = await process_folder(folder_to_process)
        
        if success:
            save_processed_folder(folder_to_process)
            current_idx += 1  # Move to the next folder
        else:
            # If failed, wait and retry the same folder
            print(f"Failed to process {folder_to_process}. Retrying in {CHECK_INTERVAL} seconds...")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main()) 