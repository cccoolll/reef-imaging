import os
import asyncio
import aiohttp
from typing import List, Tuple, Dict, Any, Optional, Callable, Set
import time
from datetime import datetime
import traceback
from asyncio import Queue

from .core import HyphaConnection, UploadRecord, Config

class ArtifactUploader:
    """Handles uploading files to Hypha artifact manager"""
    
    def __init__(self, 
                 artifact_alias: str, 
                 record_file: str,
                 connection: Optional[HyphaConnection] = None,
                 concurrency_limit: int = Config.CONCURRENCY_LIMIT,
                 client_id: str = "reef-client"):
        """Initialize the uploader with the artifact alias and record file"""
        self.artifact_alias = artifact_alias
        self.upload_record = UploadRecord(record_file)
        self.connection = connection or HyphaConnection()
        self.concurrency_limit = concurrency_limit
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.client_id = client_id
    
    async def ensure_connected(self) -> bool:
        """Ensure we have a connection to the artifact manager. Return True if connection is successful."""
        try:
            if not self.connection.artifact_manager:
                await self.connection.connect(client_id=self.client_id)
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def extract_date_time_from_path(self, path: str) -> str:
        """Extract date and time from folder name"""
        folder_name = os.path.basename(path)
        parts = folder_name.split('_')
        if len(parts) >= 3:
            # Format: 20250410-fucci-time-lapse-scan_2025-04-10_13-50-7.762411
            return parts[1] + '_' + parts[2].split('.')[0]  # Returns: 2025-04-10_13-50-7
        return folder_name  # Fallback to full folder name if format doesn't match
    
    async def upload_single_file(
        self,
        local_file: str,
        relative_path: str,
        session: aiohttp.ClientSession,
        max_retries: int = Config.MAX_RETRIES,
        retry_delay: int = Config.INITIAL_RETRY_DELAY
    ) -> bool:
        """Upload a single file to the artifact manager with immediate retries for connection issues"""
        # Skip if file was already uploaded
        if self.upload_record.is_uploaded(relative_path):
            return True

        retries = 0
        current_delay = retry_delay
        
        while retries < max_retries:
            try:
                async with self.semaphore:
                    # Make sure we're connected before attempting upload
                    connected = await self.ensure_connected()
                    if not connected:
                        print(f"Failed to ensure connection for {relative_path}")
                        retries += 1
                        await asyncio.sleep(current_delay)
                        # Reset connection for the next attempt
                        await self.connection.disconnect()
                        current_delay = min(current_delay * 2, 60)  # Exponential backoff up to 60s
                        continue

                    # 1) Get the presigned URL with timeout
                    try:
                        put_url = await asyncio.wait_for(
                            self.connection.artifact_manager.put_file(self.artifact_alias, file_path=relative_path),
                            timeout=Config.CONNECTION_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        print(f"Timeout getting presigned URL for {relative_path}")
                        # Reset connection and retry
                        await self.connection.disconnect()
                        retries += 1
                        await asyncio.sleep(current_delay)
                        current_delay = min(current_delay * 2, 60)
                        continue
                    except Exception as e:
                        print(f"Error getting presigned URL for {relative_path}: {str(e)}")
                        await self.connection.disconnect()
                        retries += 1
                        await asyncio.sleep(current_delay)
                        current_delay = min(current_delay * 2, 60)
                        continue

                    # 2) Use aiohttp session to PUT the data with timeout
                    try:
                        with open(local_file, "rb") as file_data:
                            async with asyncio.timeout(Config.UPLOAD_TIMEOUT):
                                async with session.put(put_url, data=file_data) as resp:
                                    if resp.status != 200:
                                        raise RuntimeError(
                                            f"File upload failed for {local_file}, status={resp.status}"
                                        )
                    except asyncio.TimeoutError:
                        print(f"Upload timed out after {Config.UPLOAD_TIMEOUT} seconds for {relative_path}")
                        retries += 1
                        await asyncio.sleep(current_delay)
                        current_delay = min(current_delay * 2, 60)
                        continue
                    except Exception as e:
                        print(f"Error uploading {relative_path}: {str(e)}")
                        retries += 1
                        await asyncio.sleep(current_delay)
                        current_delay = min(current_delay * 2, 60)
                        continue

                    # 3) Record successful upload
                    self.upload_record.mark_uploaded(relative_path)

                    print(
                        f"Uploaded file: {relative_path} ({self.upload_record.completed_files}/{self.upload_record.total_files})"
                    )
                    return True

            except Exception as e:
                print(f"Unexpected error uploading {relative_path}: {str(e)}")
                traceback.print_exc()
                retries += 1
                await asyncio.sleep(current_delay)
                current_delay = min(current_delay * 2, 60)
                # Try to reset connection on any error
                try:
                    await self.connection.disconnect()
                except:
                    pass

        print(f"Failed to upload {relative_path} after {max_retries} attempts")
        return False
    
    async def retry_upload_with_new_connection(
        self,
        local_file: str, 
        relative_path: str,
        retries: int = Config.MAX_RETRIES,
        initial_delay: int = Config.INITIAL_RETRY_DELAY
    ) -> bool:
        """
        Retry uploading a file with a completely new connection.
        This is a more aggressive retry strategy for consistently failing files.
        """
        retry_count = 0
        retry_delay = initial_delay
        
        while retry_count < retries:
            if retry_count > 0:
                print(f"Deep retry {retry_count}/{retries} for {relative_path}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, Config.MAX_RETRY_DELAY)
            
            # Create a fresh connection for this specific retry
            temp_connection = HyphaConnection()
            temp_uploader = None
            
            try:
                # Connect with a timeout
                await asyncio.wait_for(
                    temp_connection.connect(client_id=self.client_id),
                    timeout=Config.CONNECTION_TIMEOUT
                )
                
                # Create a temporary uploader with this connection
                temp_uploader = ArtifactUploader(
                    self.artifact_alias,
                    self.upload_record.record_file,
                    connection=temp_connection,
                    concurrency_limit=1  # Use 1 for individual retries
                )
                
                # Try the upload with a fresh session
                async with aiohttp.ClientSession() as session:
                    success = await temp_uploader.upload_single_file(
                        local_file,
                        relative_path,
                        session,
                        max_retries=10 
                    )
                    
                    if success:
                        # Refresh our record from disk
                        self.upload_record.load()
                        return True
                
            except asyncio.TimeoutError:
                print(f"Connection timed out during deep retry {retry_count} for {relative_path}")
            except Exception as e:
                print(f"Error during deep retry {retry_count} for {relative_path}: {str(e)}")
            finally:
                # Clean up the temporary connection
                if temp_connection:
                    try:
                        await temp_connection.disconnect()
                    except:
                        pass
            
            retry_count += 1
        
        print(f"Failed to upload {relative_path} after {retries} deep retry attempts")
        return False
    
    async def process_batch(
        self,
        batch: List[Tuple[str, str]],
        session: aiohttp.ClientSession,
        file_retries: int = 3
    ) -> List[Tuple[str, str]]:
        """Process a batch of files and return list of failures for deeper retry"""
        tasks = []
        for local_file, relative_path in batch:
            if self.upload_record.is_uploaded(relative_path):
                continue
                
            task = asyncio.create_task(
                self.upload_single_file(
                    local_file,
                    relative_path,
                    session,
                    max_retries=file_retries
                )
            )
            tasks.append((task, local_file, relative_path))
        
        # Wait for all tasks to complete, collect failures
        failed_uploads = []
        for task, local_file, relative_path in tasks:
            try:
                success = await task
                if not success:
                    failed_uploads.append((local_file, relative_path))
            except Exception as e:
                print(f"Unhandled exception for {relative_path}: {str(e)}")
                # Cancel the task if it's still running
                if not task.done():
                    task.cancel()
                failed_uploads.append((local_file, relative_path))
        
        return failed_uploads
    
    async def upload_files_in_batches(self, to_upload: List[Tuple[str, str]], batch_size: int = 5) -> bool:
        """Upload files in batches using a queue for better concurrency."""
        queue = Queue()
        for item in to_upload:
            await queue.put(item)

        async def worker(session: aiohttp.ClientSession):
            while True:
                try:
                    local_file, relative_path = await asyncio.wait_for(queue.get(), timeout=1.0) # Wait briefly for an item
                except asyncio.TimeoutError:
                    # Queue might be empty or temporarily starved, check if we should exit
                    if queue.empty() and all(t.done() or t.cancelled() for t in tasks if t is not asyncio.current_task()):
                        break # Exit if queue is empty and other workers are finishing/done
                    continue # Otherwise, continue waiting
                
                try:
                    # Use the session passed to the worker
                    success = await self.upload_single_file(local_file, relative_path, session)
                    if not success:
                        print(f"Failed to upload file: {local_file}, re-queuing for retry after delay.")
                        # Add a small delay before re-queuing
                        await asyncio.sleep(2) 
                        await queue.put((local_file, relative_path))
                except Exception as e:
                    print(f"Error in worker for {local_file}: {e}. Re-queuing.")
                    # Also re-queue on unexpected errors within the worker task
                    await asyncio.sleep(2) 
                    await queue.put((local_file, relative_path))
                finally:
                    queue.task_done()

        # Create a fixed number of worker tasks, each with its own session
        tasks = []
        async with aiohttp.ClientSession() as shared_session: # Create one session shared by workers
            for _ in range(min(batch_size, self.concurrency_limit)): # Limit workers by concurrency_limit too
                tasks.append(asyncio.create_task(worker(shared_session)))

            # Wait for the queue to be fully processed
            await queue.join()

            # Cancel all worker tasks
            for task in tasks:
                task.cancel()

            # Check if there are any remaining items in the queue
            if not queue.empty():
                print("Some files failed to upload after multiple attempts.")
                return False

        return True

    async def upload_zarr_files(self, file_paths: List[str]) -> bool:
        """Upload files to the artifact manager using batch processing."""
        to_upload = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                for root, _, files in os.walk(file_path):
                    for file in files:
                        local_file = os.path.join(root, file)
                        rel_path = os.path.relpath(local_file, file_path)
                        # Get basename without .zarr extension
                        base_name = os.path.basename(file_path)
                        if base_name.endswith('.zarr'):
                            base_name = base_name[:-5]  # Remove the .zarr extension
                        relative_path = os.path.join(base_name, rel_path)
                        to_upload.append((local_file, relative_path))
            else:
                local_file = file_path
                # Get basename without .zarr extension for individual files too
                relative_path = os.path.basename(file_path)
                if relative_path.endswith('.zarr'):
                    relative_path = relative_path[:-5]  # Remove the .zarr extension
                to_upload.append((local_file, relative_path))

        self.upload_record.set_total_files(len(to_upload))

        # Use batch processing for uploads
        success = await self.upload_files_in_batches(to_upload)

        if success:
            self.upload_record.save()

        return success

    async def upload_treatment_data(self, source_dirs: List[str]) -> bool:
        """Upload treatment data files to the artifact manager using batch processing."""
        to_upload = []
        for source_dir in source_dirs:
            folder_name = self.extract_date_time_from_path(source_dir)
            for root, _, files in os.walk(source_dir):
                for file in files:
                    local_file = os.path.join(root, file)
                    rel_path = os.path.relpath(local_file, source_dir)
                    relative_path = os.path.join(folder_name, rel_path)
                    to_upload.append((local_file, relative_path))

        self.upload_record.set_total_files(len(to_upload))

        # Use batch processing for uploads
        success = await self.upload_files_in_batches(to_upload)

        if success:
            self.upload_record.save()

        return success

async def upload_zarr_example() -> None:
    """Example of uploading zarr files"""
    # Original zarr paths with .zarr extension
    ORIGINAL_ZARR_PATHS = [
        "/media/reef/harddisk/test_stitch_zarr/2025-04-10_13-50-7.zarr",
        "/media/reef/harddisk/test_stitch_zarr/2025-04-10_14-50-7.zarr"
    ]
    
    uploader = ArtifactUploader(
        artifact_alias="image-map-20250410-treatment",
        record_file="zarr_upload_record.json"
    )
    
    success = await uploader.upload_zarr_files(ORIGINAL_ZARR_PATHS)
    
    if success:
        # Commit the dataset if all files were uploaded successfully
        from .gallery_manager import GalleryManager
        gallery_manager = GalleryManager()
        await gallery_manager.commit_dataset("image-map-20250410-treatment")
        await gallery_manager.connection.disconnect()

async def upload_treatment_example() -> None:
    """Example of uploading treatment data"""
    # List of source directories to upload
    SOURCE_DIRS = [
        "/media/reef/harddisk/20250410-fucci-time-lapse-scan_2025-04-10_13-50-7.762411",
        "/media/reef/harddisk/20250410-fucci-time-lapse-scan_2025-04-10_14-50-7.948398"
    ]
    
    uploader = ArtifactUploader(
        artifact_alias="20250410-treatment",
        record_file="treatment_upload_record.json"
    )
    
    success = await uploader.upload_treatment_data(SOURCE_DIRS)
    
    if success:
        # Commit the dataset if all files were uploaded successfully
        from .gallery_manager import GalleryManager
        gallery_manager = GalleryManager()
        await gallery_manager.commit_dataset("20250410-treatment")
        await gallery_manager.connection.disconnect()

if __name__ == "__main__":
    # Choose which example to run
    asyncio.run(upload_zarr_example())
    # asyncio.run(upload_treatment_example()) 