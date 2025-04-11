import os
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from hypha_rpc import connect_to_server
from dotenv import load_dotenv
import glob
import shutil
import tempfile
import time
from typing import Optional

# Load environment variables
load_dotenv()
SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_TOKEN = os.getenv("REEF_WORKSPACE_TOKEN")
DATASET_ALIAS = "20250410-treatment"
# List of source directories to upload
SOURCE_DIRS = [
    "/media/reef/harddisk/20250410-fucci-time-lapse-scan_2025-04-10_13-50-7.762411",
    "/media/reef/harddisk/20250410-fucci-time-lapse-scan_2025-04-10_14-50-7.948398"
]
CONCURRENCY_LIMIT = 5  # Max number of concurrent uploads
UPLOAD_RECORD_FILE = "treatment_upload_record-20250410.json"  # File to track uploaded files
MAX_RETRIES = 30  # Maximum number of retry attempts
INITIAL_RETRY_DELAY = 5  # Initial retry delay in seconds
MAX_RETRY_DELAY = 60  # Maximum retry delay in seconds

def load_upload_record():
    """Load the record of previously uploaded files"""
    if os.path.exists(UPLOAD_RECORD_FILE):
        with open(UPLOAD_RECORD_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
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

def extract_date_time_from_path(path):
    """Extract date and time from folder name"""
    folder_name = os.path.basename(path)
    parts = folder_name.split('_')
    if len(parts) >= 3:
        # Format: 20250410-fucci-time-lapse-scan_2025-04-10_13-50-7.762411
        return parts[1] + '_' + parts[2].split('.')[0]  # Returns: 2025-04-10_13-50-7
    return folder_name  # Fallback to full folder name if format doesn't match

async def get_artifact_manager() -> tuple:
    """Get a new connection to the artifact manager"""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": WORKSPACE_TOKEN}
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    return api, artifact_manager

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
    Requests a presigned URL from artifact_manager, then does an async PUT to upload the file.
    Implements retry logic with exponential backoff.
    """
    # Skip if file was already uploaded
    if relative_path in upload_record["uploaded_files"]:
        return

    retry_count = 0
    retry_delay = INITIAL_RETRY_DELAY

    while retry_count < MAX_RETRIES:
        try:
            async with semaphore:
                # 1) Get the presigned URL
                put_url = await artifact_manager.put_file(
                    artifact_alias, file_path=relative_path
                )

                # 2) Use aiohttp session to PUT the data
                async with session.put(put_url, data=open(local_file, "rb")) as resp:
                    if resp.status != 200:
                        raise RuntimeError(
                            f"File upload failed for {local_file}, status={resp.status}"
                        )

                # 3) Record successful upload
                upload_record["uploaded_files"].add(relative_path)
                upload_record["completed_files"] += 1

                # 4) Save progress periodically (every 10 files)
                if upload_record["completed_files"] % 10 == 0:
                    save_upload_record(upload_record)

                print(
                    f"Uploaded file: {relative_path} ({upload_record['completed_files']}/{upload_record['total_files']})"
                )
                return  # Success, exit the retry loop

        except Exception as e:
            retry_count += 1
            if retry_count == MAX_RETRIES:
                print(f"Failed to upload {relative_path} after {MAX_RETRIES} attempts. Error: {str(e)}")
                raise

            print(f"Retry {retry_count}/{MAX_RETRIES} for {relative_path}. Error: {str(e)}")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)  # Exponential backoff with max limit

async def main():
    # Load upload record
    upload_record = load_upload_record()
    if isinstance(upload_record["uploaded_files"], list):
        upload_record["uploaded_files"] = set(upload_record["uploaded_files"])

    # 0) Connect to Artifact Manager
    api, artifact_manager = await get_artifact_manager()

    # 1) Prepare a list of (local_file, relative_path) to upload
    to_upload = []
    
    # Process each source directory
    for source_dir in SOURCE_DIRS:
        # Extract date time string to use as folder name
        folder_name = extract_date_time_from_path(source_dir)
        print(f"Processing directory: {source_dir} -> {folder_name}")
        
        # Walk through the directory structure
        for root, _, files in os.walk(source_dir):
            for file in files:
                local_file = os.path.join(root, file)
                
                # Calculate the relative path within the source directory
                rel_path = os.path.relpath(local_file, source_dir)
                
                # Create the new path with the folder name
                relative_path = os.path.join(folder_name, rel_path)
                
                to_upload.append((local_file, relative_path))

    # Update total files count
    upload_record["total_files"] = len(to_upload)
    save_upload_record(upload_record)

    # 2) Create tasks to upload each file in parallel, with concurrency limit
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []

    async with aiohttp.ClientSession() as session:
        for local_file, relative_path in to_upload:
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
            tasks.append(task)

        # 3) Run tasks concurrently
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Error during upload: {str(e)}")
            print("Attempting to reset connection and continue...")
            # Reset connection
            await api.close()
            api, artifact_manager = await get_artifact_manager()
            # Retry failed uploads
            failed_tasks = [t for t in tasks if not t.done() or t.exception()]
            if failed_tasks:
                print(f"Retrying {len(failed_tasks)} failed uploads...")
                await asyncio.gather(*failed_tasks)

    # 4) Save final record and commit the dataset
    save_upload_record(upload_record)
    await artifact_manager.commit(DATASET_ALIAS)
    print("Dataset committed successfully.")
    print(
        f"Total files uploaded: {upload_record['completed_files']}/{upload_record['total_files']}"
    )

if __name__ == "__main__":
    asyncio.run(main()) 