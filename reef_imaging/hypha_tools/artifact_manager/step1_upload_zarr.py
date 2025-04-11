# step1_upload_tiles_async.py
import os
import json
from datetime import datetime
import asyncio
import aiohttp
from hypha_rpc import connect_to_server
from dotenv import load_dotenv
import time

load_dotenv()  # Loads environment variables, e.g. AGENT_LENS_WORKSPACE_TOKEN

SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_TOKEN = os.getenv("REEF_WORKSPACE_TOKEN")
# Original zarr paths with .zarr extension
ORIGINAL_ZARR_PATHS = [
    "/media/reef/harddisk/test_stitch_zarr/2025-04-10_13-50-7.zarr",
    "/media/reef/harddisk/test_stitch_zarr/2025-04-10_14-50-7.zarr"
]
ARTIFACT_ALIAS = "image-map-20250410-treatment"
CONCURRENCY_LIMIT = 5  # Max number of concurrent uploads
UPLOAD_RECORD_FILE = "upload_record.json"  # File to track uploaded files
MAX_RETRIES = 300  # Maximum number of retry attempts
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
    Single attempt to upload a file. No internal retries.
    """
    # Skip if file was already uploaded
    if relative_path in upload_record["uploaded_files"]:
        print(f"Skipping already uploaded file: {relative_path}")
        return True

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
        try:
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
        
        except Exception as e:
            print(f"Connection error during retry {retry_count} for {relative_path}: {str(e)}")
        
        # Cleanup connection before retrying
        try:
            await api.close()
        except:
            pass
            
        retry_count += 1
    
    print(f"Failed to upload {relative_path} after {retries} attempts")
    return False


async def main():
    # Load upload record
    upload_record = load_upload_record()
    if isinstance(upload_record["uploaded_files"], list):
        upload_record["uploaded_files"] = set(upload_record["uploaded_files"])

    # 0) Connect to Artifact Manager
    api, artifact_manager = await get_artifact_manager()

    # 1) Prepare a list of (local_file, relative_path) to upload
    to_upload = []
    
    for zarr_path in ORIGINAL_ZARR_PATHS:
        # Extract the folder name without .zarr extension to use as the top-level directory
        folder_name = os.path.basename(zarr_path).replace('.zarr', '')
        
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

    # 2) First attempt to upload files in parallel
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    initial_tasks = []
    failed_uploads = []

    async with aiohttp.ClientSession() as session:
        # Create tasks for initial attempt
        for local_file, relative_path in to_upload:
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
            initial_tasks.append((task, local_file, relative_path))
        
        # Wait for all tasks to complete and collect failures
        for task, local_file, relative_path in initial_tasks:
            try:
                success = await task
                if not success:
                    failed_uploads.append((local_file, relative_path))
            except Exception as e:
                print(f"Exception in task for {relative_path}: {str(e)}")
                failed_uploads.append((local_file, relative_path))

    # Close the initial connection
    await api.close()
    
    # 3) Retry failed uploads one by one with fresh connections
    print(f"First pass completed. Retrying {len(failed_uploads)} failed uploads...")
    
    retry_tasks = []
    for local_file, relative_path in failed_uploads:
        task = asyncio.create_task(
            retry_upload_with_new_connections(
                local_file, 
                relative_path, 
                upload_record
            )
        )
        retry_tasks.append(task)
    
    # Wait for all retry tasks to complete
    await asyncio.gather(*retry_tasks)

    # 4) Save final record and get a fresh connection to commit
    save_upload_record(upload_record)
    
    # Final connection for commit
    api, artifact_manager = await get_artifact_manager()
    try:
        await artifact_manager.commit(ARTIFACT_ALIAS)
        print("Dataset committed successfully.")
    except Exception as e:
        print(f"Error committing dataset: {str(e)}")
    finally:
        await api.close()
        
    print(
        f"Total files uploaded: {upload_record['completed_files']}/{upload_record['total_files']}"
    )


if __name__ == "__main__":
    asyncio.run(main())

# Created/Modified files during execution:
print(["step1_upload_zarr.py", "upload_record.json"])
