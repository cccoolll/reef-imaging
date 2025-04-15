import os
import asyncio
import json
import aiohttp
import sys
import time
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

# Constants
BASE_DIR = "/media/reef/harddisk"
EXPERIMENT_ID = "20250410-fucci-time-lapse-scan"
UPLOAD_RECORD_FILE = "treatment_upload_progress.txt"
UPLOAD_TRACKER_FILE = "treatment_upload_record.json"
CHECK_INTERVAL = 60  # Check for new folders every 60 seconds

# Load environment variables
load_dotenv()
SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_TOKEN = os.getenv("REEF_WORKSPACE_TOKEN")
DATASET_ALIAS = "20250410-treatment"

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


def load_upload_record():
    """Load the record of previously uploaded files"""
    if os.path.exists(UPLOAD_TRACKER_FILE):
        with open(UPLOAD_TRACKER_FILE, "r", encoding="utf-8") as f:
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

    with open(UPLOAD_TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(record_copy, f, indent=2)


def extract_date_time_from_path(path):
    """Extract date and time from folder name"""
    folder_name = os.path.basename(path)
    parts = folder_name.split('_')
    if len(parts) >= 3:
        # Format: 20250410-fucci-time-lapse-scan_2025-04-10_13-50-7.762411
        return parts[1] + '_' + parts[2].split('.')[0]  # Returns: 2025-04-10_13-50-7
    return folder_name  # Fallback to full folder name if format doesn't match


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
    """Process a batch of files with a timeout wrapper"""
    tasks = []
    for local_file, relative_path in batch:
        if relative_path in upload_record["uploaded_files"]:
            continue

        task = asyncio.create_task(
            upload_single_file(
                artifact_manager,
                DATASET_ALIAS,
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

    return failed_uploads


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
                        DATASET_ALIAS,
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


async def process_folder(folder_path):
    """Process a single folder by uploading all files inside."""
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return False
    
    # Load upload record
    upload_record = load_upload_record()
    
    # Initialize per-folder tracking
    folder_total_files = 0
    folder_completed_files = 0
    
    # 0) Put the dataset in staging mode
    try:
        api, artifact_manager = await get_artifact_manager()
        
        try:
            # Read the current manifest
            dataset = await artifact_manager.read(
                artifact_id=DATASET_ALIAS,
                silent=True
            )
            
            # Put the dataset in staging mode
            print(f"Putting dataset {DATASET_ALIAS} in staging mode...")
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
    
    # Extract date time string to use as folder name
    folder_name = extract_date_time_from_path(folder_path)
    print(f"Processing directory: {folder_path} -> {folder_name}")
    
    # Walk through the directory structure
    for root, _, files in os.walk(folder_path):
        for file in files:
            local_file = os.path.join(root, file)
            
            # Calculate the relative path within the source directory
            rel_path = os.path.relpath(local_file, folder_path)
            
            # Create the new path with the folder name
            relative_path = os.path.join(folder_name, rel_path)
            
            to_upload.append((local_file, relative_path))
            folder_total_files += 1

    # Update total files count while preserving existing completed files
    upload_record["total_files"] = len(to_upload) + upload_record.get("total_files", 0)
    save_upload_record(upload_record)
    
    print(f"\nStarting upload for folder: {folder_name}")
    print(f"Files to upload in this folder: {folder_total_files}")
    
    # Connect to Artifact Manager
    api, artifact_manager = await get_artifact_manager()

    # 2) First attempt to upload files in parallel, but in smaller batches
    BATCH_SIZE = 20  # Process files in smaller batches to limit potential timeouts
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
                artifact_manager.commit(DATASET_ALIAS),
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
        f"Folder upload complete: {folder_completed_files}/{folder_total_files} files uploaded in {folder_name}\n"
        f"Total files uploaded across all folders: {upload_record['completed_files']}/{upload_record['total_files']}"
    )
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