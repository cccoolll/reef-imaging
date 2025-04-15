import os
import asyncio
import aiohttp
from typing import List, Tuple, Dict, Any, Optional, Callable, Set
import time
from datetime import datetime

from .core import HyphaConnection, UploadRecord, Config

class ArtifactUploader:
    """Handles uploading files to Hypha artifact manager"""
    
    def __init__(self, 
                 artifact_alias: str, 
                 record_file: str,
                 connection: Optional[HyphaConnection] = None,
                 concurrency_limit: int = Config.CONCURRENCY_LIMIT):
        """Initialize the uploader with the artifact alias and record file"""
        self.artifact_alias = artifact_alias
        self.upload_record = UploadRecord(record_file)
        self.connection = connection or HyphaConnection()
        self.concurrency_limit = concurrency_limit
        self.semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def ensure_connected(self) -> None:
        """Ensure we have a connection to the artifact manager"""
        if not self.connection.artifact_manager:
            await self.connection.connect()
    
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
        retries: int = Config.MAX_RETRIES,
        initial_delay: int = Config.INITIAL_RETRY_DELAY
    ) -> bool:
        """Upload a single file to the artifact manager with retry mechanism"""
        # Skip if file was already uploaded
        if self.upload_record.is_uploaded(relative_path):
            return True

        retry_count = 0
        retry_delay = initial_delay

        while retry_count < retries:
            try:
                async with self.semaphore:
                    await self.ensure_connected()

                    # 1) Get the presigned URL with timeout
                    try:
                        put_url = await asyncio.wait_for(
                            self.connection.artifact_manager.put_file(self.artifact_alias, file_path=relative_path),
                            timeout=Config.CONNECTION_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        print(f"Timeout getting presigned URL for {relative_path}")
                        retry_count += 1
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, Config.MAX_RETRY_DELAY)
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
                        retry_count += 1
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, Config.MAX_RETRY_DELAY)
                        continue

                    # 3) Record successful upload
                    self.upload_record.mark_uploaded(relative_path)

                    print(
                        f"Uploaded file: {relative_path} ({self.upload_record.completed_files}/{self.upload_record.total_files})"
                    )
                    return True

            except Exception as e:
                print(f"Error uploading {relative_path}: {str(e)}")
                retry_count += 1
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, Config.MAX_RETRY_DELAY)

        print(f"Failed to upload {relative_path} after {retries} attempts")
        return False
    
    async def retry_upload_with_new_connections(
        self,
        local_file: str, 
        relative_path: str,
        retries: int = Config.MAX_RETRIES,
        initial_delay: int = Config.INITIAL_RETRY_DELAY
    ) -> bool:
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
                retry_delay = min(retry_delay * 2, Config.MAX_RETRY_DELAY)  # Exponential backoff
            
            # Get a fresh connection for each retry
            conn = HyphaConnection()
            try:
                # Create a new connection with timeout
                await conn.connect()
                temp_uploader = ArtifactUploader(
                    self.artifact_alias, 
                    self.upload_record.record_file,
                    connection=conn, 
                    concurrency_limit=1  # Use 1 for individual retries
                )
                
                # Attempt the upload with a fresh session
                async with aiohttp.ClientSession() as session:
                    success = await temp_uploader.upload_single_file(
                        local_file,
                        relative_path,
                        session
                    )
                    
                    if success:
                        # Update our record to match the temp uploader's record
                        self.upload_record.load()
                        return True
            
            except asyncio.TimeoutError:
                print(f"Connection timed out during retry {retry_count} for {relative_path}")
            except Exception as e:
                print(f"Connection error during retry {retry_count} for {relative_path}: {str(e)}")
            
            # Cleanup connection before retrying
            try:
                await conn.close()
            except:
                pass
                
            retry_count += 1
        
        print(f"Failed to upload {relative_path} after {retries} attempts")
        return False
    
    async def process_batch(
        self,
        batch: List[Tuple[str, str]],
        session: aiohttp.ClientSession
    ) -> List[Tuple[str, str]]:
        """Process a batch of files with a timeout wrapper"""
        tasks = []
        for local_file, relative_path in batch:
            if self.upload_record.is_uploaded(relative_path):
                continue
                
            task = asyncio.create_task(
                self.upload_single_file(
                    local_file,
                    relative_path,
                    session
                )
            )
            tasks.append((task, local_file, relative_path))
        
        # Wait for all tasks to complete with a timeout, collect failures
        failed_uploads = []
        for task, local_file, relative_path in tasks:
            try:
                # Set a reasonable timeout for the entire operation
                success = await asyncio.wait_for(task, timeout=Config.UPLOAD_TIMEOUT * 2)
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
    
    async def upload_zarr_files(self, zarr_paths: List[str]) -> bool:
        """Upload zarr files to the artifact manager"""
        # 1) Prepare a list of (local_file, relative_path) to upload
        to_upload = []
        
        for zarr_path in zarr_paths:
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
        self.upload_record.set_total_files(len(to_upload))
        
        # 2) First attempt to upload files in parallel, but in smaller batches
        BATCH_SIZE = 20  # Process files in smaller batches to limit potential timeouts
        batches = [to_upload[i:i + BATCH_SIZE] for i in range(0, len(to_upload), BATCH_SIZE)]
        failed_uploads = []
        
        # Process each batch with a fresh session
        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} files)")
            try:
                # Use a fresh session for each batch
                async with aiohttp.ClientSession() as session:
                    batch_failures = await self.process_batch(batch, session)
                    failed_uploads.extend(batch_failures)
                    
                # Save progress after each batch
                self.upload_record.save()
                
            except Exception as e:
                print(f"Error processing batch {i+1}: {str(e)}")
                # If a batch fails entirely, add all its files to the failed list
                for local_file, relative_path in batch:
                    if not self.upload_record.is_uploaded(relative_path):
                        failed_uploads.append((local_file, relative_path))

        # Close the initial connection
        try:
            await self.connection.close()
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
                    self.retry_upload_with_new_connections(
                        local_file, 
                        relative_path
                    )
                )
                retry_tasks.append(task)
            
            # Wait for this batch of retries to complete
            await asyncio.gather(*retry_tasks)
            
            # Save progress after each retry batch
            self.upload_record.save()
            
        # 4) Save final record
        self.upload_record.save()
        
        print(
            f"Total files uploaded: {self.upload_record.completed_files}/{self.upload_record.total_files}"
        )
        
        # Return success if all files were uploaded
        return self.upload_record.completed_files == self.upload_record.total_files
    
    async def upload_treatment_data(self, source_dirs: List[str]) -> bool:
        """Upload treatment data files to the artifact manager"""
        # 1) Prepare a list of (local_file, relative_path) to upload
        to_upload = []
        
        # Process each source directory
        for source_dir in source_dirs:
            # Extract date time string to use as folder name
            folder_name = self.extract_date_time_from_path(source_dir)
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
        self.upload_record.set_total_files(len(to_upload))
        
        # 2) First attempt to upload files in parallel, but in smaller batches
        BATCH_SIZE = 20  # Process files in smaller batches to limit potential timeouts
        batches = [to_upload[i:i + BATCH_SIZE] for i in range(0, len(to_upload), BATCH_SIZE)]
        failed_uploads = []
        
        # Process each batch with a fresh session
        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} files)")
            try:
                # Use a fresh session for each batch
                async with aiohttp.ClientSession() as session:
                    batch_failures = await self.process_batch(batch, session)
                    failed_uploads.extend(batch_failures)
                    
                # Save progress after each batch
                self.upload_record.save()
                
            except Exception as e:
                print(f"Error processing batch {i+1}: {str(e)}")
                # If a batch fails entirely, add all its files to the failed list
                for local_file, relative_path in batch:
                    if not self.upload_record.is_uploaded(relative_path):
                        failed_uploads.append((local_file, relative_path))
        
        # Close the initial connection
        try:
            await self.connection.close()
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
                    self.retry_upload_with_new_connections(
                        local_file, 
                        relative_path
                    )
                )
                retry_tasks.append(task)
            
            # Wait for this batch of retries to complete
            await asyncio.gather(*retry_tasks)
            
            # Save progress after each retry batch
            self.upload_record.save()
            
        # 4) Save final record
        self.upload_record.save()
        
        print(
            f"Total files uploaded: {self.upload_record.completed_files}/{self.upload_record.total_files}"
        )
        
        # Return success if all files were uploaded
        return self.upload_record.completed_files == self.upload_record.total_files

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
        await gallery_manager.connection.close()

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
        await gallery_manager.connection.close()

if __name__ == "__main__":
    # Choose which example to run
    asyncio.run(upload_zarr_example())
    # asyncio.run(upload_treatment_example()) 