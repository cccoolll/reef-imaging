import os
import asyncio
import aiohttp
from typing import List, Tuple, Dict, Any, Optional, Callable, Set
import time
from datetime import datetime
import traceback
from asyncio import Queue
import random

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
    
    async def connect_with_retry(self, client_id=None, max_retries=300, base_delay=5, max_delay=60):
        """Connect to Hypha with exponential backoff and retry."""
        if client_id:
            self.client_id = client_id
        
        client_already_exists_count = 0
        retry_count = 0
        connect_task = None
        
        while retry_count < max_retries:
            try:
                # Always ensure we're fully disconnected before attempting to connect
                await self.connection.disconnect()
                
                # Wait longer if we've seen "Client already exists" errors
                if client_already_exists_count > 0:
                    # Exponential backoff with jitter for client conflicts
                    delay = min(max_delay, base_delay * (2 ** client_already_exists_count)) + random.uniform(1, 5)
                    print(f"Waiting {delay:.1f}s before reconnect attempt (client conflict detected)")
                    await asyncio.sleep(delay)
                
                # Create connection task with timeout
                print(f"Attempting connection to {self.connection.server_url} with client_id: {self.client_id}")
                connect_task = asyncio.create_task(self.connection.connect(client_id=self.client_id))
                await asyncio.wait_for(connect_task, timeout=Config.CONNECTION_TIMEOUT)
                print("Connection established successfully")
                return True
                
            except asyncio.TimeoutError:
                retry_count += 1
                print(f"Connection attempt timed out after {Config.CONNECTION_TIMEOUT}s (attempt {retry_count}/{max_retries})")
                # Clean up the task
                if connect_task and not connect_task.done():
                    connect_task.cancel()
                    # Wait a moment for cancellation to process
                    try:
                        await asyncio.wait_for(asyncio.shield(connect_task), timeout=1)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                
                # Use standard retry backoff
                delay = min(max_delay, base_delay * (2 ** min(retry_count, 5)))
                print(f"Will retry in {delay:.1f}s")
                await asyncio.sleep(delay)
                
            except Exception as e:
                retry_count += 1
                err_msg = str(e)
                print(f"Connection error: {err_msg}")
                
                # Clean up the task if it exists and is not done
                if connect_task and not connect_task.done():
                    connect_task.cancel()
                
                if "Client already exists" in err_msg:
                    client_already_exists_count += 1
                    print(f"Client ID conflict detected. Ensuring only one instance is running or use unique client IDs.")
                    
                    # More aggressive cleanup and longer wait for reconnection
                    try:
                        await self.connection.disconnect()
                    except:
                        pass
                    
                    # Special handling for deep client conflicts
                    if client_already_exists_count >= 3:
                        print(f"Persistent client conflict. Waiting longer for server-side cleanup...")
                        # Wait longer when we have persistent conflicts
                        await asyncio.sleep(client_already_exists_count * 10)
                else:
                    # For other errors, use standard retry backoff
                    delay = min(max_delay, base_delay * (2 ** min(retry_count, 5)))
                    print(f"Will retry in {delay:.1f}s (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(delay)
                
                if retry_count >= max_retries:
                    print(f"Failed to connect after {max_retries} attempts")
                    return False
                
        return False
    
    async def ensure_connected(self) -> bool:
        """Ensure we have a connection to the artifact manager. Return True if connection is successful."""
        try:
            if not self.connection.artifact_manager:
                # Create a task for the connection so we can cancel it if needed
                connect_task = asyncio.create_task(self.connection.connect(client_id=self.client_id))
                try:
                    # Wait for the connection with a timeout
                    await asyncio.wait_for(connect_task, timeout=Config.CONNECTION_TIMEOUT)
                except asyncio.TimeoutError:
                    # Cancel the task on timeout
                    if not connect_task.done():
                        connect_task.cancel()
                    print(f"Connection attempt timed out after {Config.CONNECTION_TIMEOUT} seconds")
                    return False
                except Exception as e:
                    # Cancel the task on error
                    if not connect_task.done():
                        connect_task.cancel()
                    print(f"Connection failed: {e}")
                    return False
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
        connection_reset = False
        file_size = os.path.getsize(local_file)
        
        # Get optimal timeout based on file size (larger files need more time)
        upload_timeout = min(Config.UPLOAD_TIMEOUT, max(60, file_size // (500 * 1024)))  # 500KB/s minimum rate
        
        while retries < max_retries:
            try:
                async with self.semaphore:
                    # Make sure we're connected before attempting upload
                    connected = await self.ensure_connected()
                    if not connected:
                        print(f"Failed to ensure connection for {relative_path}")
                        retries += 1
                        await asyncio.sleep(current_delay)
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
                        retries += 1
                        await asyncio.sleep(current_delay)
                        current_delay = min(current_delay * 2, 60)
                        continue
                    except Exception as e:
                        print(f"Error getting presigned URL for {relative_path}: {str(e)}")
                        retries += 1
                        await asyncio.sleep(current_delay)
                        current_delay = min(current_delay * 2, 60)
                        continue

                    # 2) Use aiohttp session to PUT the data with timeout and optimized settings
                    try:
                        headers = {
                            'Content-Type': 'application/octet-stream',
                            'Content-Length': str(file_size)
                        }
                        
                        # Use different strategies based on file size
                        if file_size > 10 * 1024 * 1024:  # For files > 10MB
                            # Stream large files in chunks
                            with open(local_file, "rb") as file_data:
                                async with asyncio.timeout(upload_timeout):
                                    async with session.put(
                                        put_url, 
                                        data=file_data,
                                        headers=headers,
                                        chunked=True,
                                        timeout=aiohttp.ClientTimeout(total=upload_timeout)
                                    ) as resp:
                                        if resp.status != 200:
                                            raise RuntimeError(
                                                f"File upload failed for {local_file}, status={resp.status}"
                                            )
                        else:
                            # Load smaller files into memory for faster upload
                            with open(local_file, "rb") as file_data:
                                data = file_data.read()
                                async with asyncio.timeout(upload_timeout):
                                    async with session.put(
                                        put_url, 
                                        data=data,
                                        headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=upload_timeout)
                                    ) as resp:
                                        if resp.status != 200:
                                            raise RuntimeError(
                                                f"File upload failed for {local_file}, status={resp.status}"
                                            )
                    except asyncio.TimeoutError:
                        print(f"Upload timed out after {upload_timeout} seconds for {relative_path}")
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

            # Reset connection only after 5 failed attempts
            if retries >= 5 and not connection_reset:
                try:
                    print(f"Resetting connection after {retries} failed attempts for file {relative_path}")
                    # Ensure any running connect tasks are properly canceled
                    # We don't have direct access to the tasks from here, so we rely on the disconnect method
                    
                    # Perform a full disconnection
                    await self.connection.disconnect()
                    
                    # Wait briefly to allow complete cleanup
                    await asyncio.sleep(1)
                    
                    # Mark that we reset the connection for this file
                    connection_reset = True
                    
                    print(f"Connection reset completed for file {relative_path}")
                except Exception as e:
                    print(f"Error resetting connection: {e}")
                    # Continue with retries even if reset fails

        print(f"Failed to upload {relative_path} after {max_retries} attempts")
        return False
    
    async def deep_retry_upload(
        self,
        local_file: str, 
        relative_path: str,
        retries: int = Config.MAX_RETRIES,
        initial_delay: int = Config.INITIAL_RETRY_DELAY
    ) -> bool:
        """
        Deep retry for uploading a file that failed during batch process.
        Uses existing connection with incremental backoff strategy.
        """
        retry_count = 0
        retry_delay = initial_delay
        connect_task = None
        
        while retry_count < retries:
            if retry_count > 0:
                print(f"Deep retry {retry_count}/{retries} for {relative_path}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, Config.MAX_RETRY_DELAY)
            
            try:
                # Check if we need to reset connection after 5 failed attempts
                if retry_count >= 5:
                    print(f"File {relative_path} failed over {retry_count} times, resetting connection...")
                    # Cancel any existing connection task first
                    if connect_task and not connect_task.done():
                        connect_task.cancel()
                        # Wait a moment for cancellation to process
                        try:
                            await asyncio.wait_for(asyncio.shield(connect_task), timeout=1)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                    
                    # Then disconnect
                    await self.connection.disconnect()
                    await asyncio.sleep(2)  # Brief pause to ensure clean disconnect
                
                # Ensure existing connection is valid
                if not self.connection.artifact_manager:
                    print(f"Connection not established, attempting to connect with client_id: {self.client_id}")
                    
                    try:
                        # Create a proper connection task that we can cancel if needed
                        connect_task = asyncio.create_task(self.connection.connect(client_id=self.client_id))
                        await asyncio.wait_for(connect_task, timeout=Config.CONNECTION_TIMEOUT)
                        print("Connection established successfully")
                    except asyncio.TimeoutError:
                        print(f"Connection attempt timed out during deep retry {retry_count}")
                        if connect_task and not connect_task.done():
                            connect_task.cancel()
                        retry_count += 1
                        continue
                    except Exception as e:
                        print(f"Connection error during deep retry {retry_count}: {str(e)}")
                        if connect_task and not connect_task.done():
                            connect_task.cancel()
                        retry_count += 1
                        continue
                
                # Try the upload with a new session
                async with aiohttp.ClientSession() as session:
                    success = await self.upload_single_file(
                        local_file,
                        relative_path,
                        session,
                        max_retries=5
                    )
                    
                    if success:
                        # Update record from disk
                        self.upload_record.load()
                        return True
            
            except Exception as e:
                print(f"Error during deep retry {retry_count} for {relative_path}: {str(e)}")
                traceback.print_exc()
            
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
    
    async def upload_files_in_batches(self, to_upload: List[Tuple[str, str]], batch_size: int = 20) -> bool:
        """
        Upload files in batches using a queue for better concurrency.
        Optimized with:
        1. TCP connection pooling
        2. Separate queue for presigned URLs
        3. More efficient worker management
        4. Dynamic batch size adjustment
        """
        # Create queues for files and presigned URLs
        file_queue = Queue()
        url_queue = Queue()
        
        # Add all files to the queue
        for item in to_upload:
            await file_queue.put(item)
        
        # Track files being processed
        in_progress = set()
        failed_files = set()
        
        # Create TCP connector with optimized settings
        connector = aiohttp.TCPConnector(
            limit=Config.CONNECTION_POOL_SIZE,
            force_close=False,
            enable_cleanup_closed=True,
            keepalive_timeout=60
        )
        
        # Create semaphore for limiting concurrent operations
        url_semaphore = asyncio.Semaphore(Config.CONCURRENCY_LIMIT)

        async def url_worker():
            """Worker to get presigned URLs in batches"""
            while True:
                try:
                    # Get a batch of files to process
                    batch = []
                    batch_size = min(Config.URL_BATCH_SIZE, file_queue.qsize())
                    if batch_size == 0:
                        if file_queue.empty() and not in_progress:
                            break
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Fill the batch
                    for _ in range(batch_size):
                        try:
                            item = file_queue.get_nowait()
                            batch.append(item)
                            in_progress.add(item)
                        except asyncio.QueueEmpty:
                            break
                            
                    if not batch:
                        await asyncio.sleep(0.1)
                        continue
                        
                    # Ensure connection before getting URLs
                    async with url_semaphore:
                        connected = await self.ensure_connected()
                        if not connected:
                            # Put items back in queue if connection fails
                            for item in batch:
                                await file_queue.put(item)
                                in_progress.remove(item)
                            await asyncio.sleep(1)
                            continue
                            
                        # Process each file in the batch to get URL
                        for local_file, relative_path in batch:
                            # Skip if already uploaded
                            if self.upload_record.is_uploaded(relative_path):
                                in_progress.remove((local_file, relative_path))
                                file_queue.task_done()
                                continue
                                
                            try:
                                # Get presigned URL with timeout
                                put_url = await asyncio.wait_for(
                                    self.connection.artifact_manager.put_file(
                                        self.artifact_alias, 
                                        file_path=relative_path
                                    ),
                                    timeout=Config.CONNECTION_TIMEOUT
                                )
                                
                                # Put URL in queue for upload workers
                                await url_queue.put((local_file, relative_path, put_url))
                                
                            except Exception as e:
                                print(f"Error getting URL for {relative_path}: {str(e)}")
                                # Re-queue the file for retry with backoff
                                file_queue.task_done()
                                in_progress.remove((local_file, relative_path))
                                
                                # Add to failed files to retry later with exponential backoff
                                failed_files.add((local_file, relative_path))
                                
                except Exception as e:
                    print(f"Error in URL worker: {str(e)}")
                    await asyncio.sleep(1)
        
        async def upload_worker(session: aiohttp.ClientSession):
            """Worker to upload files using presigned URLs"""
            while True:
                try:
                    # Get a file and its presigned URL
                    try:
                        local_file, relative_path, put_url = await asyncio.wait_for(
                            url_queue.get(), 
                            timeout=0.5
                        )
                    except asyncio.TimeoutError:
                        # Check if we should exit
                        if url_queue.empty() and file_queue.empty() and not in_progress:
                            break
                        continue
                    
                    # Attempt upload with optimized settings
                    success = False
                    retries = 0
                    
                    while retries < 3 and not success:  # Limit retries per worker
                        try:
                            file_size = os.path.getsize(local_file)
                            buffer_size = min(file_size, 1024 * 1024)  # 1MB buffer for large files
                            
                            with open(local_file, "rb") as file_data:
                                headers = {
                                    'Content-Type': 'application/octet-stream',
                                    'Content-Length': str(file_size)
                                }
                                
                                async with asyncio.timeout(Config.UPLOAD_TIMEOUT):
                                    if file_size > 10 * 1024 * 1024:  # For files > 10MB
                                        # Stream file in chunks for large files
                                        async with session.put(
                                            put_url, 
                                            data=file_data, 
                                            headers=headers,
                                            chunked=True,
                                            timeout=aiohttp.ClientTimeout(total=Config.UPLOAD_TIMEOUT)
                                        ) as resp:
                                            if resp.status == 200:
                                                success = True
                                            else:
                                                print(f"Upload failed with status {resp.status} for {relative_path}")
                                    else:
                                        # Small file upload
                                        data = file_data.read()
                                        async with session.put(
                                            put_url, 
                                            data=data,
                                            headers=headers,
                                            timeout=aiohttp.ClientTimeout(total=Config.UPLOAD_TIMEOUT)
                                        ) as resp:
                                            if resp.status == 200:
                                                success = True
                                            else:
                                                print(f"Upload failed with status {resp.status} for {relative_path}")
                                
                                if success:
                                    # Record successful upload
                                    self.upload_record.mark_uploaded(relative_path)
                                    print(f"Uploaded file: {relative_path} ({self.upload_record.completed_files}/{self.upload_record.total_files})")
                                    
                        except Exception as e:
                            retries += 1
                            print(f"Upload error for {relative_path}: {str(e)} (retry {retries}/3)")
                            await asyncio.sleep(1)
                    
                    if not success:
                        # If still failed after retries, add to failed files
                        failed_files.add((local_file, relative_path))
                    
                    # Mark as done in URL queue and remove from in_progress
                    url_queue.task_done()
                    file_queue.task_done()
                    in_progress.remove((local_file, relative_path))
                    
                except Exception as e:
                    print(f"Error in upload worker: {str(e)}")
                    await asyncio.sleep(1)
        
        # Start worker tasks with optimized session
        async with aiohttp.ClientSession(connector=connector) as session:
            # Calculate number of workers based on system resources
            num_url_workers = min(4, batch_size)  # Fewer URL workers to avoid overwhelming the API
            num_upload_workers = min(Config.MAX_WORKERS, batch_size)
            
            # Create worker tasks
            url_workers = [asyncio.create_task(url_worker()) for _ in range(num_url_workers)]
            upload_workers = [asyncio.create_task(upload_worker(session)) for _ in range(num_upload_workers)]
            
            # Wait for queues to be processed
            try:
                await file_queue.join()
                await url_queue.join()
            except Exception as e:
                print(f"Error waiting for queues: {str(e)}")
            
            # Cancel worker tasks
            for task in url_workers + upload_workers:
                task.cancel()
            
            # Handle any failed files with deep retry
            if failed_files:
                print(f"{len(failed_files)} files failed during normal upload, attempting deep retry...")
                retry_success = 0
                connection_task = None
                
                try:
                    # Ensure we have a working connection before starting deep retries
                    # First, properly disconnect and clean up any existing tasks
                    await self.connection.disconnect()
                    await asyncio.sleep(2)  # Wait for server to clean up
                    
                    # Reconnect with our main connection
                    print("Creating fresh connection for deep retries")
                    try:
                        # Create a new connection task we can track and cancel if needed
                        connection_task = asyncio.create_task(self.connection.connect(client_id=self.client_id))
                        await asyncio.wait_for(connection_task, timeout=Config.CONNECTION_TIMEOUT)
                        connect_success = True
                    except asyncio.TimeoutError:
                        print("Connection timeout during deep retry reconnection")
                        if connection_task and not connection_task.done():
                            connection_task.cancel()
                        connect_success = False
                    except Exception as e:
                        print(f"Connection error during deep retry reconnection: {e}")
                        if connection_task and not connection_task.done():
                            connection_task.cancel()
                        connect_success = False
                    
                    if connect_success:
                        # Try deep retry for each failed file using the existing connection
                        for local_file, relative_path in failed_files:
                            if await self.deep_retry_upload(local_file, relative_path):
                                retry_success += 1
                    
                    print(f"Deep retry recovered {retry_success}/{len(failed_files)} files")
                    
                    # Return failure if any files still failed
                    if retry_success < len(failed_files):
                        return False
                except Exception as e:
                    print(f"Error during deep retry process: {e}")
                    # Ensure connection task is canceled if there's an error
                    if connection_task and not connection_task.done():
                        connection_task.cancel()
                finally:
                    # Ensure we always cleanup any hanging tasks
                    if connection_task and not connection_task.done():
                        connection_task.cancel()
        
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