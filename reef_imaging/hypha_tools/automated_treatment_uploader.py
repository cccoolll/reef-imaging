import os
import asyncio
import json
import sys
import time
import random  # Add this import
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
# Timeout and retry settings
CONNECTION_TIMEOUT = 60  # Timeout for connection operations in seconds
OPERATION_TIMEOUT = 120  # Timeout for Hypha operations in seconds
MAX_RETRIES = 300  # Maximum retries for operations
# Optimized concurrency settings
BATCH_SIZE = 30  # Batch size for file uploads (increased from default)
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


async def connect_with_timeout(connection, client_id, timeout=CONNECTION_TIMEOUT, max_retries=MAX_RETRIES):
    """Connect to Hypha with timeout and retry logic."""
    retry_count = 0
    client_conflict_count = 0
    
    while retry_count < max_retries:
        try:
            # Always ensure disconnection before attempting to connect
            await connection.disconnect()
            
            # Add longer delay if we've seen client conflicts
            if client_conflict_count > 0:
                conflict_delay = min(60, 10 * client_conflict_count) + random.uniform(1, 5)
                print(f"Waiting {conflict_delay:.1f}s before reconnection due to client conflict")
                await asyncio.sleep(conflict_delay)
            
            # Create task with timeout
            print(f"Attempting connection to Hypha with client_id: {client_id}")
            connect_task = asyncio.create_task(connection.connect(client_id=client_id))
            await asyncio.wait_for(connect_task, timeout=timeout)
            print("Connection established successfully")
            return True
            
        except asyncio.TimeoutError:
            retry_count += 1
            print(f"Connection attempt timed out after {timeout}s (attempt {retry_count}/{max_retries})")
            # Clean up the task and connection
            if 'connect_task' in locals() and not connect_task.done():
                connect_task.cancel()
            await connection.disconnect()
            if retry_count < max_retries:
                # Wait before retrying
                await asyncio.sleep(5)
                
        except Exception as e:
            retry_count += 1
            err_msg = str(e)
            print(f"Connection error: {err_msg} (attempt {retry_count}/{max_retries})")
            
            # Handle "Client already exists" error specifically
            if "Client already exists" in err_msg:
                client_conflict_count += 1
                print("Client ID conflict detected. Ensure only one instance is running or use unique client IDs.")
                
                # Clean up connection more aggressively
                if 'connect_task' in locals() and not connect_task.done():
                    connect_task.cancel()
                
                try:
                    await connection.disconnect()
                except:
                    pass
                
                # Add a longer delay for client conflicts to allow server to clean up
                # Increase delay with each conflict
                conflict_delay = min(60, 10 * client_conflict_count) + random.uniform(1, 5)
                print(f"Waiting {conflict_delay:.1f}s for server-side client cleanup")
                await asyncio.sleep(conflict_delay)
            else:
                # For other errors, perform normal cleanup
                await connection.disconnect()
                if retry_count < max_retries:
                    await asyncio.sleep(5)
    
    print("Failed to connect after multiple attempts")
    return False


async def process_folder(folder_path):
    """Process a single folder by uploading all files inside."""
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return False
    
    # Create uploader instance with optimized concurrency
    uploader = ArtifactUploader(
        artifact_alias=DATASET_ALIAS,
        record_file=UPLOAD_TRACKER_FILE,
        client_id=client_id,
        concurrency_limit=CONCURRENCY_LIMIT
    )
    
    # Create a fresh connection with TCP optimizations
    connection = HyphaConnection()
    
    try:
        # Connect to Hypha with timeout and retry
        connect_success = await connect_with_timeout(connection, client_id)
        if not connect_success:
            print("Failed to establish connection to Hypha")
            return False
        
        # Assign connection to uploader
        uploader.connection = connection
        
        # Put dataset in staging mode
        staging_success = False
        retry_count = 0
        
        while not staging_success and retry_count < MAX_RETRIES:
            try:
                print(f"Putting dataset {DATASET_ALIAS} in staging mode...")
                artifact_manager = connection.artifact_manager
                
                dataset_manifest = {
                    "name": "Full treatment dataset 20250410",
                    "description": "The Full treatment dataset from 20250410",
                }
                
                # Put the dataset in staging mode with timeout
                await asyncio.wait_for(
                    artifact_manager.edit(
                        artifact_id=DATASET_ALIAS,
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
                
                # Reset connection on timeout
                await connection.disconnect()
                if retry_count < MAX_RETRIES:
                    connect_success = await connect_with_timeout(connection, client_id)
                    if not connect_success:
                        return False
                    uploader.connection = connection
                    await asyncio.sleep(5)
            
            except Exception as e:
                retry_count += 1
                print(f"Error putting dataset in staging mode: {e} (attempt {retry_count}/{MAX_RETRIES})")
                
                if "not found" in str(e).lower():
                    print("Dataset not found. It may need to be created first.")
                    return False
                
                # Reset connection on error
                await connection.disconnect()
                if retry_count < MAX_RETRIES:
                    connect_success = await connect_with_timeout(connection, client_id)
                    if not connect_success:
                        return False
                    uploader.connection = connection
                    await asyncio.sleep(5)
        
        if not staging_success:
            print("Failed to put dataset in staging mode after multiple attempts")
            return False
        
        # Extract folder name for organizing uploads
        folder_name = os.path.basename(folder_path)
        extracted_name = uploader.extract_date_time_from_path(folder_path)
        print(f"Processing directory: {folder_path} -> {extracted_name}")
        
        # Upload the folder contents with optimized batch size
        # Modified to use the optimized batch size parameter
        to_upload = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                local_file = os.path.join(root, file)
                rel_path = os.path.relpath(local_file, folder_path)
                relative_path = os.path.join(extracted_name, rel_path)
                to_upload.append((local_file, relative_path))
        
        print(f"Found {len(to_upload)} files to upload")
        uploader.upload_record.set_total_files(len(to_upload))
        
        # Use the optimized upload_files_in_batches with higher batch size
        success = await uploader.upload_files_in_batches(to_upload, batch_size=BATCH_SIZE)
        
        # Commit the dataset
        if success:
            commit_success = False
            commit_attempts = 0
            
            while not commit_success and commit_attempts < 5:
                try:
                    # Refresh connection before commit
                    await connection.disconnect()
                    connect_success = await connect_with_timeout(connection, client_id)
                    if not connect_success:
                        print("Failed to reconnect before commit")
                        if commit_attempts < 4:  # Try again if we have attempts left
                            commit_attempts += 1
                            await asyncio.sleep(5)
                            continue
                        else:
                            return False
                    
                    # Update uploader connection
                    uploader.connection = connection
                    
                    # Commit the dataset with timeout
                    print(f"Committing dataset (attempt {commit_attempts + 1}/5)...")
                    await asyncio.wait_for(
                        connection.artifact_manager.commit(DATASET_ALIAS),
                        timeout=OPERATION_TIMEOUT
                    )
                    print("Dataset committed successfully.")
                    commit_success = True
                    
                except asyncio.TimeoutError:
                    commit_attempts += 1
                    print(f"Commit operation timed out (attempt {commit_attempts}/5)")
                    await connection.disconnect()
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    commit_attempts += 1
                    print(f"Error committing dataset (attempt {commit_attempts}/5): {str(e)}")
                    await connection.disconnect()
                    await asyncio.sleep(5)
            
            if not commit_success:
                print("WARNING: Failed to commit the dataset after multiple attempts.")
                return False
                
        return success
        
    except Exception as e:
        print(f"Error processing folder: {e}")
        return False
    finally:
        # Clean up connection
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
            #delete treatment_upload_record.json file
            if os.path.exists(UPLOAD_TRACKER_FILE):
                os.remove(UPLOAD_TRACKER_FILE)
        else:
            # If failed, retry after a delay
            print(f"Failed to process {folder_to_process}. Retrying in {CHECK_INTERVAL} seconds...")
            await asyncio.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main()) 