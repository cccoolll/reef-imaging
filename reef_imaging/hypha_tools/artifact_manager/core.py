import os
import asyncio
import aiohttp
from hypha_rpc import connect_to_server
from dotenv import load_dotenv
import json
import random
from datetime import datetime
from typing import Dict, Set, Any, Tuple, Optional, List, Union

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the artifact manager"""
    SERVER_URL = "https://hypha.aicell.io"
    WORKSPACE_TOKEN = os.getenv("REEF_WORKSPACE_TOKEN")
    CONCURRENCY_LIMIT = 25  # Max number of concurrent uploads (increased from 10)
    MAX_RETRIES = 300  # Maximum number of retry attempts
    INITIAL_RETRY_DELAY = 5  # Initial retry delay in seconds
    MAX_RETRY_DELAY = 60  # Maximum retry delay in seconds
    CONNECTION_TIMEOUT = 30  # Timeout for API connections in seconds
    UPLOAD_TIMEOUT = 120  # Timeout for file uploads in seconds (increased from 60)
    URL_BATCH_SIZE = 30  # Number of presigned URLs to request at once
    MAX_WORKERS = 20  # Maximum number of worker tasks
    MIN_URL_WORKERS = 6  # Minimum number of URL workers
    CONNECTION_POOL_SIZE = 100  # TCP connection pool size
    MAX_FAILED_FILES = 10  # Maximum number of failed files before resetting connection
    MAX_COMMIT_ATTEMPTS = 15  # Maximum number of commit attempts
    MAX_COMMIT_DELAY = 220  # Maximum delay between commit attempts in seconds
class UploadRecord:
    """Manages the record of uploaded files"""
    
    def __init__(self, record_file: str):
        self.record_file = record_file
        self.uploaded_files: Set[str] = set()
        self.last_update: Optional[str] = None
        self.total_files: int = 0
        self.completed_files: int = 0
        self.load()
    
    def load(self) -> None:
        """Load the record of previously uploaded files"""
        if os.path.exists(self.record_file):
            with open(self.record_file, "r", encoding="utf-8") as f:
                record = json.load(f)
                self.uploaded_files = set(record.get("uploaded_files", []))
                self.last_update = record.get("last_update")
                self.total_files = record.get("total_files", 0)
                self.completed_files = record.get("completed_files", 0)
    
    def save(self) -> None:
        """Save the record of uploaded files"""
        # Convert set to list for JSON serialization
        record = {
            "uploaded_files": list(self.uploaded_files),
            "last_update": datetime.now().isoformat(),
            "total_files": self.total_files,
            "completed_files": self.completed_files
        }
        
        with open(self.record_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
    
    def is_uploaded(self, relative_path: str) -> bool:
        """Check if a file has been uploaded"""
        return relative_path in self.uploaded_files
    
    def mark_uploaded(self, relative_path: str) -> None:
        """Mark a file as uploaded"""
        self.uploaded_files.add(relative_path)
        self.completed_files += 1
        
        # Save progress periodically (every 10 files)
        if self.completed_files % 10 == 0:
            self.save()
    
    def set_total_files(self, total: int) -> None:
        """Set the total number of files to upload"""
        self.total_files = total
        self.save()

class HyphaConnection:
    """Manages connections to the Hypha server"""
    
    def __init__(self, server_url: str = Config.SERVER_URL, token: str = Config.WORKSPACE_TOKEN):
        self.server_url = server_url
        self.token = token
        self.api = None
        self.artifact_manager = None
    
    async def connect(self, timeout: int = Config.CONNECTION_TIMEOUT, client_id: str = "reef-client") -> None:
        """Connect to the Hypha server with robust error handling"""
        # Always attempt to disconnect first to clear any lingering state
        await self.disconnect()
        
        try:
            print(f"Attempting connection to {self.server_url} with client_id: {client_id}")
            self.api = await asyncio.wait_for(
                connect_to_server({
                    "client_id": client_id, 
                    "server_url": self.server_url, 
                    "token": self.token,
                }),
                timeout=timeout
            )
            print("Connection established, getting artifact manager...")
            self.artifact_manager = await asyncio.wait_for(
                self.api.get_service("public/artifact-manager"),
                timeout=timeout
            )
            print("Connected successfully to Hypha and artifact manager")
        except asyncio.TimeoutError:
            print(f"Connection attempt timed out after {timeout} seconds")
            # Ensure cleanup even on timeout during connection or service retrieval
            await self.disconnect() 
            raise
        except Exception as e:
            # Catch specific errors if possible, e.g., check 'Client already exists'
            error_msg = str(e)
            print(f"Connection error: {error_msg}")
            if "Client already exists" in error_msg:
                 print("Client ID conflict detected. Ensure only one instance is running or use unique client IDs.")
            # Ensure cleanup on any connection error
            await self.disconnect() 
            raise # Re-raise the exception after cleanup

        
    async def connect_with_retry(self, client_id=None, max_retries=300, base_delay=5, max_delay=180):
        """Connect to Hypha with exponential backoff and retry."""
        if client_id:
            self.client_id = client_id
        
        client_already_exists_count = 0
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Always ensure we're fully disconnected before attempting to connect
                await self.disconnect()
                
                # Cancel any existing connection task
                if self.connection_task and not self.connection_task.done():
                    self.connection_task.cancel()
                    # Wait a moment for cancellation to process
                    try:
                        await asyncio.wait_for(asyncio.shield(self.connection_task), timeout=1)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    self.connection_task = None
                
                # Wait longer if we've seen "Client already exists" errors
                if client_already_exists_count > 0:
                    # Exponential backoff with jitter for client conflicts
                    delay = min(max_delay, base_delay * (2 ** client_already_exists_count)) + random.uniform(1, 5)
                    print(f"Waiting {delay:.1f}s before reconnect attempt (client conflict detected)")
                    await asyncio.sleep(delay)
                
                # Create connection task with timeout
                print(f"Attempting connection to {self.server_url} with client_id: {self.client_id}")
                self.connection_task = asyncio.create_task(self.connect(client_id=self.client_id))
                await asyncio.wait_for(self.connection_task, timeout=Config.CONNECTION_TIMEOUT)
                print("Connection established successfully")
                return True
                
            except asyncio.TimeoutError:
                retry_count += 1
                print(f"Connection attempt timed out after {Config.CONNECTION_TIMEOUT}s (attempt {retry_count}/{max_retries})")
                # Clean up the task
                if self.connection_task and not self.connection_task.done():
                    self.connection_task.cancel()
                    # Wait a moment for cancellation to process
                    try:
                        await asyncio.wait_for(asyncio.shield(self.connection_task), timeout=1)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    self.connection_task = None
                
                # Use standard retry backoff
                delay = min(max_delay, base_delay * (2 ** min(retry_count, 5)))
                print(f"Will retry in {delay:.1f}s")
                await asyncio.sleep(delay)
                
            except Exception as e:
                retry_count += 1
                err_msg = str(e)
                print(f"Connection error: {err_msg}")
                
                # Clean up the task if it exists and is not done
                if self.connection_task and not self.connection_task.done():
                    self.connection_task.cancel()
                    self.connection_task = None
                
                if "Client already exists" in err_msg:
                    client_already_exists_count += 1
                    print(f"Client ID conflict detected. Ensuring only one instance is running or use unique client IDs.")
                    
                    # More aggressive cleanup and longer wait for reconnection
                    try:
                        await self.disconnect()
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
    
    async def disconnect(self, timeout: int = 5) -> None:
        """Disconnect the connection to the Hypha server gracefully."""
        if self.api is not None:
            print("Disconnecting from Hypha server...")
            try:
                await asyncio.wait_for(self.api.disconnect(), timeout=timeout)
                print("Hypha API disconnected successfully.")
            except asyncio.TimeoutError:
                print(f"Hypha disconnect timed out after {timeout} seconds.")
            except Exception as e:
                print(f"Error during Hypha API disconnection: {e}")
            finally:
                # Always reset state variables after attempting disconnect
                self.api = None
                self.artifact_manager = None
        else:
            # Ensure state is clean even if api was already None
            self.api = None
            self.artifact_manager = None
            # print("Already disconnected or connection not established.") # Optional: uncomment for more verbose logging

    async def reconnect(self, timeout: int = Config.CONNECTION_TIMEOUT, client_id: str = "reef-client") -> None:
        """Reconnect to the Hypha server"""
        print("Attempting to reconnect...")
        await self.disconnect() # Ensure clean state before reconnecting
        await self.connect(timeout=timeout, client_id=client_id)
