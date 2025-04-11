# step1_upload_tiles_async.py
import os
import json
from datetime import datetime
import asyncio
import aiohttp
from hypha_rpc import connect_to_server
from dotenv import load_dotenv

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
    Requests a presigned URL from artifact_manager, then does an async PUT to upload the file.
    Implements retry logic with exponential backoff.
    """
    # Skip if file was already uploaded
    if relative_path in upload_record["uploaded_files"]:
        print(f"Skipping already uploaded file: {relative_path}")
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

    # 2) Create tasks to upload each file in parallel, with concurrency limit
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []

    async with aiohttp.ClientSession() as session:
        for local_file, relative_path in to_upload:
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
    await artifact_manager.commit(ARTIFACT_ALIAS)
    print("Dataset committed successfully.")
    print(
        f"Total files uploaded: {upload_record['completed_files']}/{upload_record['total_files']}"
    )


if __name__ == "__main__":
    asyncio.run(main())

# Created/Modified files during execution:
print(["step1_upload_zarr.py", "upload_record.json"])
