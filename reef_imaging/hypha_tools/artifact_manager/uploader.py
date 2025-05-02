import os
import asyncio
import aiohttp
import requests
from typing import List, Tuple, Dict, Any, Optional
import time
from datetime import datetime
import traceback
import json
import zipfile
import tempfile
import shutil

from .core import HyphaConnection, UploadRecord, Config

class ArtifactUploader:
    """Handles uploading files to Hypha artifact manager"""
    
    def __init__(self, 
                 artifact_alias: str, 
                 record_file: str,
                 connection: Optional[HyphaConnection] = None,
                 client_id: str = "reef-client"):
        """Initialize the uploader with the artifact alias and record file"""
        self.artifact_alias = artifact_alias
        self.upload_record = UploadRecord(record_file)
        self.connection = connection or HyphaConnection()
        self.client_id = client_id
        self.connection_task = None  # Track connection task at class level
        self.last_progress_time = None

    async def connect_with_retry(self, client_id=None, max_retries=5, base_delay=5):
        """Connect to Hypha with simple retry."""
        if client_id:
            self.client_id = client_id
        
        for attempt in range(max_retries):
            try:
                # Clean disconnect first
                await self.connection.disconnect()
                
                # Connect with client ID
                await self.connection.connect(client_id=self.client_id)
                print("Connection established successfully")
                return True
                
            except Exception as e:
                print(f"Connection error (attempt {attempt+1}/{max_retries}): {str(e)}")
                await asyncio.sleep(base_delay)
                
        print(f"Failed to connect after {max_retries} attempts")
        return False
    
    async def ensure_connected(self) -> bool:
        """Ensure we have a connection to the artifact manager."""
        if not self.connection.artifact_manager:
            return await self.connect_with_retry()
        return True
    
    def extract_date_time_from_path(self, path: str) -> str:
        """Extract date and time from folder name"""
        folder_name = os.path.basename(path)
        parts = folder_name.split('_')
        if len(parts) >= 3:
            # Format: 20250410-fucci-time-lapse-scan_2025-04-10_13-50-7.762411
            return parts[1] + '_' + parts[2].split('.')[0]  # Returns: 2025-04-10_13-50-7
        return folder_name  # Fallback to full folder name if format doesn't match
    
    async def upload_single_file(self, local_file: str, relative_path: str) -> bool:
        """Upload a single file to the artifact manager using the direct approach from the tutorial."""
        # Skip if file was already uploaded
        if self.upload_record.is_uploaded(relative_path):
            print(f"File {relative_path} already uploaded, skipping")
            return True

        file_size = os.path.getsize(local_file)
        print(f"Starting upload of {relative_path} ({file_size/1024/1024:.2f} MB)")
        
        # Retry loop
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Make sure we're connected
                connected = await self.ensure_connected()
                if not connected:
                    print(f"Failed to connect, retrying...")
                    await asyncio.sleep(5)
                    continue

                # Get the presigned URL - follow tutorial approach
                put_url = await self.connection.artifact_manager.put_file(
                    self.artifact_alias, 
                    file_path=relative_path
                )
                
                # Upload using requests - direct approach from tutorial
                print(f"Uploading {relative_path} using direct PUT request")
                with open(local_file, "rb") as file_data:
                    response = requests.put(
                        put_url, 
                        data=file_data,
                        headers={'Content-Type': 'application/octet-stream'}
                    )
                    if not response.ok:
                        raise RuntimeError(f"File upload failed for {local_file}, status={response.status_code}")
                
                # Record successful upload
                self.upload_record.mark_uploaded(relative_path)
                print(f"Successfully uploaded {relative_path} ({self.upload_record.completed_files}/{self.upload_record.total_files})")
                self.last_progress_time = time.time()
                return True

            except Exception as e:
                print(f"Error uploading {relative_path} (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    # Reset connection on error
                    await self.connection.disconnect()
                    await asyncio.sleep(5 * (attempt + 1))  # Simple backoff
                
        print(f"Failed to upload {relative_path} after {max_retries} attempts")
        return False

    async def upload_files(self, to_upload: List[Tuple[str, str]]) -> bool:
        """Upload multiple files, one at a time - simplified without queues or complex concurrency."""
        # Set total files for progress tracking
        self.upload_record.set_total_files(len(to_upload))
        
        # Process files sequentially
        success = True
        for local_file, relative_path in to_upload:
            file_success = await self.upload_single_file(local_file, relative_path)
            if not file_success:
                success = False
                print(f"Warning: Failed to upload {relative_path}")
        
        return success

    async def zip_and_upload_folder(self, folder_path: str, relative_path: str = None, delete_zip_after: bool = True) -> bool:
        """Zip a folder and upload it as a single file."""
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist")
            return False
            
        # Use folder name for the zip filename if relative_path not provided
        if relative_path is None:
            folder_name = os.path.basename(folder_path)
            relative_path = folder_name
            
        # Ensure the relative path has .zip extension
        if not relative_path.endswith('.zip'):
            relative_path += '.zip'
            
        # Create the temporary zip file
        parent_dir = os.path.dirname(folder_path)
        temp_zip_base = os.path.basename(folder_path) + ".zip.tmp"
        temp_zip_path = os.path.join(parent_dir, temp_zip_base)
        
        # Define the synchronous zipping function
        def create_zip_file(source_path, target_zip_path):
            print(f"Creating zip file for {source_path} in background thread...")
            try:
                with zipfile.ZipFile(target_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(source_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, source_path)
                            zipf.write(file_path, arcname)
                zip_size_mb = os.path.getsize(target_zip_path) / (1024 * 1024)
                print(f"Zip file created in background thread: {target_zip_path} ({zip_size_mb:.2f} MB)")
                return True
            except Exception as e:
                print(f"Error creating zip file in background thread: {e}")
                traceback.print_exc()
                return False
        
        try:
            print(f"Starting zip file creation in background thread for {folder_path}...")
            # Run zip creation in a separate thread
            zip_success = await asyncio.to_thread(create_zip_file, folder_path, temp_zip_path)
            
            if not zip_success:
                print(f"Failed to create zip file for {folder_path}")
                return False
            
            if not os.path.exists(temp_zip_path):
                print(f"Expected zip file {temp_zip_path} doesn't exist after thread completion")
                return False
            
            # Upload the zip file
            print(f"Zip creation complete, now uploading {temp_zip_path} to {relative_path}...")
            success = await self.upload_single_file(temp_zip_path, relative_path)
            
            return success
                
        except Exception as e:
            print(f"Error during zip and upload process: {e}")
            traceback.print_exc()
            return False
        finally:
            # Clean up the temporary zip file if requested and if it exists
            if delete_zip_after and os.path.exists(temp_zip_path):
                try:
                    os.unlink(temp_zip_path)
                    print(f"Deleted temporary zip file: {temp_zip_path}")
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {temp_zip_path}: {e}")

    async def upload_zarr_files(self, file_paths: List[str]) -> bool:
        """Upload zarr files to the artifact manager - simplified."""
        for file_path in file_paths:
            print(f"Processing {file_path}...")
            
            if os.path.isdir(file_path):
                # Directory case - walk through the directory and upload files
                to_upload = []
                
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
                
                self.upload_record.set_total_files(len(to_upload))
                success = await self.upload_files(to_upload)
                
                if not success:
                    return False
            else:
                # Single file case
                local_file = file_path
                relative_path = os.path.basename(file_path)
                if relative_path.endswith('.zarr'):
                    relative_path = relative_path[:-5]  # Remove the .zarr extension
                
                success = await self.upload_single_file(local_file, relative_path)
                if not success:
                    return False
        
        return True

    async def upload_treatment_data(self, source_dirs: List[str]) -> bool:
        """Upload treatment data files - simplified approach."""
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
        success = await self.upload_files(to_upload)

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
        artifact_alias="agent-lens/image-map-20250429-treatment",
        record_file="zarr_upload_record.json"
    )
    
    success = await uploader.upload_zarr_files(ORIGINAL_ZARR_PATHS)
    
    if success:
        # Commit the dataset if all files were uploaded successfully
        from .gallery_manager import GalleryManager
        gallery_manager = GalleryManager()
        await gallery_manager.commit_dataset("")
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