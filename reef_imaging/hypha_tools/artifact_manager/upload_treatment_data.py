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

# Load environment variables
load_dotenv()
SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_TOKEN = os.getenv("REEF_WORKSPACE_TOKEN")
DATASET_ALIAS = "20250328-treatment-out-of-incubator"
SOURCE_DIR = os.path.expanduser("~/europa_disk/u2os-treatment/static/003_2025-03-28_16-20-35.283628")
CONCURRENCY_LIMIT = 5  # Max number of concurrent uploads
UPLOAD_RECORD_FILE = "treatment_upload_record.json"  # File to track uploaded files

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

def get_timepoint_folder_name(timepoint):
    """Generate the folder name based on the timepoint number"""
    base_time = datetime(2025, 3, 28, 16, 20, 35, 283628)
    # Each timepoint is 1 hour apart
    new_time = base_time + timedelta(hours=int(timepoint))
    return f"003_{new_time.strftime('%Y-%m-%d_%H-%M-%S.%f')}"

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
    """
    # Skip if file was already uploaded
    if relative_path in upload_record["uploaded_files"]:
        print(f"Skipping already uploaded file: {relative_path}")
        return

    async with semaphore:
        try:
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

        except Exception as e:
            print(f"Error uploading {relative_path}: {str(e)}")
            raise

async def main():
    # Load upload record
    upload_record = load_upload_record()
    if isinstance(upload_record["uploaded_files"], list):
        upload_record["uploaded_files"] = set(upload_record["uploaded_files"])

    # 0) Connect to Artifact Manager
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": WORKSPACE_TOKEN}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # 1) Prepare a list of (local_file, relative_path) to upload
    to_upload = []
    
    # Get all timepoint folders (0, 1, 2, etc.)
    timepoint_folders = [d for d in os.listdir(SOURCE_DIR) 
                        if os.path.isdir(os.path.join(SOURCE_DIR, d)) and d.isdigit()]
    
    # Sort timepoint folders numerically
    timepoint_folders.sort(key=int)
    
    for timepoint in timepoint_folders:
        timepoint_dir = os.path.join(SOURCE_DIR, timepoint)
        new_folder_name = get_timepoint_folder_name(timepoint)
        
        # Get all files in the timepoint folder
        for root, _, files in os.walk(timepoint_dir):
            for file in files:
                local_file = os.path.join(root, file)
                
                # Calculate the relative path within the timepoint folder
                rel_path_in_timepoint = os.path.relpath(local_file, timepoint_dir)
                
                # Create the new path with the renamed folder
                relative_path = os.path.join(new_folder_name, rel_path_in_timepoint)
                
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
        await asyncio.gather(*tasks)

    # 4) Save final record and commit the dataset
    save_upload_record(upload_record)
    await artifact_manager.commit(DATASET_ALIAS)
    print("Dataset committed successfully.")
    print(
        f"Total files uploaded: {upload_record['completed_files']}/{upload_record['total_files']}"
    )

if __name__ == "__main__":
    asyncio.run(main()) 