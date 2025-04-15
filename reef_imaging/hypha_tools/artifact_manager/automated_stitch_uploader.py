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
    """Run the stitch_zarr.py script using subprocess."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stitch_script = os.path.join(script_dir, "image_processing", "stitch_zarr.py")
    
    # Check if the script exists
    if not os.path.exists(stitch_script):
        print(f"Stitch script not found at {stitch_script}")
        return False
    
    # Clean up existing zarr files first
    existing_zarr_files = get_zarr_files()
    for zarr_file in existing_zarr_files:
        cleanup_zarr_file(zarr_file)
    
    # Create a temporary script to run stitch_zarr with the specific folder
    temp_script = os.path.join(script_dir, "temp_stitch.py")
    with open(temp_script, "w") as f:
        f.write(f"""
import sys
sys.path.insert(0, "{os.path.join(script_dir, 'image_processing')}")
import stitch_zarr

# Override the data_folders with just this folder
stitch_zarr.data_folders = ["{folder_path}"]

# Run the stitching process
stitch_zarr.main()
""")
    
    try:
        print(f"Running stitching for folder: {folder_path}")
        # Run the temporary script
        process = subprocess.run([sys.executable, temp_script], 
                                capture_output=True, 
                                text=True, 
                                check=False)
        
        if process.returncode != 0:
            print(f"Stitching failed with error code {process.returncode}")
            print(f"Error output: {process.stderr}")
            return False
        
        print(f"Stitching completed for {folder_path}")
        print(process.stdout)
        
        # Check if any zarr files were created
        zarr_files = get_zarr_files()
        if not zarr_files:
            print(f"No zarr files were created after stitching {folder_path}")
            return False
        
        return True
    except Exception as e:
        print(f"Error during stitching: {e}")
        return False
    finally:
        # Clean up temporary script
        if os.path.exists(temp_script):
            os.remove(temp_script)


async def get_artifact_manager(timeout=CONNECTION_TIMEOUT):
    """Get a new connection to the artifact manager with timeout"""
    try:
        # Direct import to avoid circular imports
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
                return False

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
        
        # Get a fresh connection for each retry
        api = None
        try:
            # Create a new connection with timeout
            api, artifact_manager = await get_artifact_manager()
            semaphore = asyncio.Semaphore(1)  # Use 1 for individual retries
            
            async with aiohttp.ClientSession() as session:
                success = await upload_single_file(
                    artifact_manager,
                    ARTIFACT_ALIAS,
                    local_file,
                    relative_path,
                    semaphore,
                    session,
                    upload_record
                )
                
                if success:
                    return True
        
        except asyncio.TimeoutError:
            print(f"Connection timed out during retry {retry_count} for {relative_path}")
        except Exception as e:
            print(f"Connection error during retry {retry_count} for {relative_path}: {str(e)}")
        
        # Cleanup connection before retrying
        if api:
            try:
                await asyncio.wait_for(api.close(), timeout=5)
            except:
                print(f"Failed to cleanly close API connection for {relative_path}")
            
        retry_count += 1
    
    print(f"Failed to upload {relative_path} after {retries} attempts")
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
    
    return failed_uploads


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
    
    # Put the dataset in staging mode
    try:
        api, artifact_manager = await get_artifact_manager()
        
        try:
            # Read the current manifest
            dataset = await artifact_manager.read(
                artifact_id=ARTIFACT_ALIAS,
                silent=True
            )
            
            # Put the dataset in staging mode
            print(f"Putting dataset {ARTIFACT_ALIAS} in staging mode...")
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
                # Close the connection
                await api.close()
                return False
        finally:
            # Close the connection
            try:
                await api.close()
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
    
    # Update total files count
    upload_record["total_files"] = len(to_upload)
    save_upload_record(upload_record)
    
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
    
    # Close the initial connection
    try:
        await asyncio.wait_for(api.close(), timeout=10)
    except:
        print("Failed to cleanly close API connection")
    
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
                    await asyncio.wait_for(api.close(), timeout=5)
                except:
                    pass
    
    if not commit_success:
        print("WARNING: Failed to commit the dataset after multiple attempts.")
        return False
    
    print(f"Total files uploaded: {upload_record['completed_files']}/{upload_record['total_files']}")
    return True


async def process_folder(folder_name: str) -> bool:
    """Process a single folder through stitching, uploading, and cleanup."""
    folder_path = os.path.join(BASE_DIR, folder_name)
    
    # Step 1: Stitch the folder
    print(f"\nStarting stitching process for folder: {folder_name}")
    stitch_success = stitch_folder(folder_path)
    if not stitch_success:
        print(f"Failed to stitch folder: {folder_name}")
        return False
    
    # Step 2: Get the stitched zarr file
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
    
    # Step 3: Upload the zarr file
    print(f"Starting upload process for zarr file: {zarr_file}")
    upload_success = await upload_zarr_file(zarr_file)
    if not upload_success:
        print(f"Failed to upload zarr file: {zarr_file}")
        return False
    
    # Step 4: Clean up the zarr file
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
        
        print(f"\nProcessing folder: {folder_to_process}")
        
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