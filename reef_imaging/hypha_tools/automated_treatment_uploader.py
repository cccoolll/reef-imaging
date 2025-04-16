import os
import asyncio
import json
import sys
import time
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

# Import shared tools from artifact_manager
from artifact_manager.core import Config, HyphaConnection
from artifact_manager.uploader import ArtifactUploader

# Constants
BASE_DIR = "/media/reef/harddisk"
EXPERIMENT_ID = "20250410-fucci-time-lapse-scan"
UPLOAD_RECORD_FILE = "treatment_upload_progress.txt"
UPLOAD_TRACKER_FILE = "treatment_upload_record.json"
CHECK_INTERVAL = 60  # Check for new folders every 60 seconds
client_id="reef-client-treatment-uploader"
# Load environment variables
load_dotenv()
DATASET_ALIAS = "20250410-treatment-full"


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
    if not os.path.exists(UPLOAD_RECORD_FILE):
        return processed
    
    with open(UPLOAD_RECORD_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line and " - " in line:
                parts = line.split(" - ", 1)
                if len(parts) == 2:
                    processed.append(parts[1])
    
    return processed


def save_processed_folder(folder_name: str):
    """Save the processed folder to the record file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(UPLOAD_RECORD_FILE, "a") as f:
        f.write(f"{timestamp} - {folder_name}\n")


async def process_folder(folder_path):
    """Process a single folder by uploading all files inside."""
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return False
    
    # Create uploader instance
    uploader = ArtifactUploader(
        artifact_alias=DATASET_ALIAS,
        record_file=UPLOAD_TRACKER_FILE,
        client_id=client_id
    )
    
    # Create a fresh connection
    connection = HyphaConnection()
    connection_task = None
    
    try:
        # Connect to Hypha
        connection_task = asyncio.create_task(connection.connect(client_id=client_id))
        await connection_task
        
        # Assign connection to uploader
        uploader.connection = connection
        
        # Put dataset in staging mode
        try:
            print(f"Putting dataset {DATASET_ALIAS} in staging mode...")
            artifact_manager = connection.artifact_manager
            
            # Read the current manifest
            dataset = await artifact_manager.read(
                artifact_id=DATASET_ALIAS,
                silent=True
            )
            
            # Put the dataset in staging mode
            await artifact_manager.edit(
                artifact_id=DATASET_ALIAS,
                manifest=dataset,  # Preserve the same manifest
                version="stage"    # Put in staging mode
            )
            print("Dataset is now in staging mode")
            
        except Exception as e:
            print(f"Error putting dataset in staging mode: {e}")
            if "not found" in str(e).lower():
                print("Dataset not found. It may need to be created first.")
                return False
        
        # Extract folder name for organizing uploads
        folder_name = os.path.basename(folder_path)
        extracted_name = uploader.extract_date_time_from_path(folder_path)
        print(f"Processing directory: {folder_path} -> {extracted_name}")
        
        # Upload the folder contents
        success = await uploader.upload_treatment_data([folder_path])
        
        # Commit the dataset
        if success:
            commit_success = False
            commit_attempts = 0
            
            while not commit_success and commit_attempts < 5:
                try:
                    # Refresh connection before commit
                    await connection.disconnect()
                    connection_task = asyncio.create_task(connection.connect(client_id=client_id))
                    await connection_task
                    
                    # Commit the dataset
                    await connection.artifact_manager.commit(DATASET_ALIAS)
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
                    connection_task = asyncio.create_task(connection.connect(client_id=client_id))
                    await connection_task
            
            if not commit_success:
                print("WARNING: Failed to commit the dataset after multiple attempts.")
                return False
                
        return success
        
    except Exception as e:
        print(f"Error processing folder: {e}")
        return False
    finally:
        # Clean up connection
        if connection_task and not connection_task.done():
            connection_task.cancel()
        await connection.disconnect()


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
    
    # Process the specified range of folders
    current_idx = start_idx
    
    while current_idx <= min(end_idx, len(all_folders) - 2):  # Don't process the last folder which might be incomplete
        folder_to_process = all_folders[current_idx]
        
        # Skip if already processed
        if folder_to_process in processed_folders:
            print(f"Folder {folder_to_process} already processed, skipping.")
            current_idx += 1
            continue
        
        print(f"\nProcessing folder: {folder_to_process}")
        folder_path = os.path.join(BASE_DIR, folder_to_process)
        
        # Process the folder
        success = await process_folder(folder_path)
        
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