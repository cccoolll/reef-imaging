import os
import asyncio
import aiohttp
from typing import List, Tuple, Dict, Any, Optional, Callable, Set
import time
from datetime import datetime
import traceback

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
    
    async def ensure_connected(self) -> bool:
        """Ensure we have a connection to the artifact manager. Return True if connection is successful."""
        try:
            if not self.connection.artifact_manager:
                await self.connection.connect()
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
                    temp_connection.connect(),
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
                        max_retries=1  # Only try once with this connection
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
        
        # 2) First attempt to upload files in parallel batches
        BATCH_SIZE = 20  # Process files in smaller batches to limit potential timeouts
        batches = [to_upload[i:i + BATCH_SIZE] for i in range(0, len(to_upload), BATCH_SIZE)]
        all_failed_uploads = []
        
        print(f"Starting upload of {len(to_upload)} files in {len(batches)} batches")
        
        # Process each batch with a fresh session
        for i, batch in enumerate(batches):
            # Ensure connection before each batch
            connection_good = await self.ensure_connected()
            if not connection_good:
                print(f"Failed to establish connection for batch {i+1}, retrying...")
                await asyncio.sleep(5)
                await self.connection.disconnect()
                connection_good = await self.ensure_connected()
                if not connection_good:
                    print(f"Still couldn't connect for batch {i+1}, adding files to retry list")
                    all_failed_uploads.extend(batch)
                    continue
                
            print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} files)")
            try:
                # Use a fresh session for each batch
                async with aiohttp.ClientSession() as session:
                    batch_failures = await self.process_batch(batch, session, file_retries=3)
                    all_failed_uploads.extend(batch_failures)
                    
                # Save progress after each batch
                self.upload_record.save()
                
            except Exception as e:
                print(f"Error processing batch {i+1}: {str(e)}")
                traceback.print_exc()
                # If a batch fails entirely, retry each file individually later
                all_failed_uploads.extend(batch)
                
                # Try to reset the connection
                try:
                    await self.connection.disconnect()
                    await asyncio.sleep(5)  # Brief pause before reconnecting
                    await self.connection.connect()
                except:
                    pass
        
        # 3) Deep retry for failed uploads, one by one with fresh connections
        if all_failed_uploads:
            print(f"First pass completed. Deep retrying {len(all_failed_uploads)} failed uploads...")
            
            # Process retries in smaller groups to avoid overloading
            retry_batch_size = 5
            for i in range(0, len(all_failed_uploads), retry_batch_size):
                retry_batch = all_failed_uploads[i:i + retry_batch_size]
                print(f"Deep retry batch {i//retry_batch_size + 1}/{(len(all_failed_uploads)-1)//retry_batch_size + 1}")
                
                # Process each file with its own dedicated connection
                retry_tasks = []
                
                for local_file, relative_path in retry_batch:
                    task = asyncio.create_task(
                        self.retry_upload_with_new_connection(
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
        
        # Count remaining failures
        remaining = len(to_upload) - self.upload_record.completed_files
        success_rate = (self.upload_record.completed_files / len(to_upload)) * 100 if len(to_upload) > 0 else 0
        
        print(
            f"Upload complete: {self.upload_record.completed_files}/{len(to_upload)} files uploaded ({success_rate:.1f}%)"
        )
        
        if remaining > 0:
            print(f"Warning: {remaining} files could not be uploaded after all retries")
            
        # Consider success if we uploaded at least 99,9% of files
        return success_rate >= 99.9
    
    async def upload_treatment_data(self, source_dirs: List[str]) -> bool:
        """Upload treatment data files to the artifact manager"""
        # Prepare a list of (local_file, relative_path) to upload
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

        # Debugging output to verify to_upload list
        print(f"Total files to upload: {len(to_upload)}")
        for local_file, relative_path in to_upload:
            print(f"Prepared for upload: {local_file} -> {relative_path}")
        
        # Update total files count
        self.upload_record.set_total_files(len(to_upload))
        
        # Process files in smaller batches to limit potential timeouts
        BATCH_SIZE = 20
        batches = [to_upload[i:i + BATCH_SIZE] for i in range(0, len(to_upload), BATCH_SIZE)]
        all_failed_uploads = []
        
        print(f"Starting upload of {len(to_upload)} files in {len(batches)} batches")
        
        # Process each batch with a fresh session
        for i, batch in enumerate(batches):
            # Ensure connection before each batch
            connection_good = await self.ensure_connected()
            if not connection_good:
                print(f"Failed to establish connection for batch {i+1}, retrying...")
                await asyncio.sleep(5)
                await self.connection.disconnect()
                connection_good = await self.ensure_connected()
                if not connection_good:
                    print(f"Still couldn't connect for batch {i+1}, adding files to retry list")
                    all_failed_uploads.extend(batch)
                    continue
                
            print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} files)")
            try:
                # Use a fresh session for each batch
                async with aiohttp.ClientSession() as session:
                    batch_failures = await self.process_batch(batch, session, file_retries=3)
                    all_failed_uploads.extend(batch_failures)
                    
                # Save progress after each batch
                self.upload_record.save()
                
            except Exception as e:
                print(f"Error processing batch {i+1}: {str(e)}")
                traceback.print_exc()
                # If a batch fails entirely, retry each file individually later
                all_failed_uploads.extend(batch)
                
                # Try to reset the connection
                try:
                    await self.connection.disconnect()
                    await asyncio.sleep(5)  # Brief pause before reconnecting
                    await self.connection.connect()
                except:
                    pass
        
        # Deep retry for failed uploads, one by one with fresh connections
        if all_failed_uploads:
            print(f"First pass completed. Deep retrying {len(all_failed_uploads)} failed uploads...")
            
            # Process retries in smaller groups to avoid overloading
            retry_batch_size = 5
            for i in range(0, len(all_failed_uploads), retry_batch_size):
                retry_batch = all_failed_uploads[i:i + retry_batch_size]
                print(f"Deep retry batch {i//retry_batch_size + 1}/{(len(all_failed_uploads)-1)//retry_batch_size + 1}")
                
                # Process each file with its own dedicated connection
                retry_tasks = []
                
                for local_file, relative_path in retry_batch:
                    task = asyncio.create_task(
                        self.retry_upload_with_new_connection(
                            local_file, 
                            relative_path
                        )
                    )
                    retry_tasks.append(task)
                
                # Wait for this batch of retries to complete
                await asyncio.gather(*retry_tasks)
                
                # Save progress after each retry batch
                self.upload_record.save()
        
        # Save final record
        self.upload_record.save()
        
        # Count remaining failures
        remaining = len(to_upload) - self.upload_record.completed_files
        success_rate = (self.upload_record.completed_files / len(to_upload)) * 100 if len(to_upload) > 0 else 0
        
        print(
            f"Upload complete: {self.upload_record.completed_files}/{len(to_upload)} files uploaded ({success_rate:.1f}%)"
        )
        
        if remaining > 0:
            print(f"Warning: {remaining} files could not be uploaded after all retries")
            
        # Consider success if we uploaded at least 99,9% of files
        return success_rate >= 99.9

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