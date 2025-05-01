import os
import asyncio
from typing import Dict, Any, Optional

from .core import HyphaConnection, Config

class GalleryManager:
    """Manages galleries and datasets in Hypha"""
    
    def __init__(self, connection: Optional[HyphaConnection] = None):
        """Initialize the gallery manager"""
        self.connection = connection or HyphaConnection()
    
    async def ensure_connected(self) -> None:
        """Ensure we have a connection to the artifact manager"""
        if not self.connection.artifact_manager:
            await self.connection.connect()
    
    async def create_gallery(self, 
                           name: str, 
                           description: str, 
                           alias: str,
                           permissions: Dict[str, str] = None) -> None:
        """Create a new gallery (collection)"""
        await self.ensure_connected()
        
        # Default permissions if none provided
        if permissions is None:
            permissions = {"*": "r+", "@": "*", "misty-teeth-42051243": "*","google-oauth2|103047988474094226050": "*"}
        
        gallery_manifest = {
            "name": name,
            "description": description,
        }
        
        await self.connection.artifact_manager.create(
            type="collection",
            alias=alias,
            manifest=gallery_manifest,
            config={"permissions": permissions},
            overwrite=True,
        )
        print(f"Gallery '{name}' created with alias '{alias}'.")
    
    async def create_dataset(self, 
                           name: str, 
                           description: str, 
                           alias: str,
                           parent_id: str,
                           version: str = "stage",
                           permissions: Dict[str, str] = None) -> None:
        """Create a new dataset within a gallery"""
        await self.ensure_connected()
        
        dataset_manifest = {
            "name": name,
            "description": description,
        }
        
        await self.connection.artifact_manager.create(
            parent_id=parent_id,
            alias=alias,
            manifest=dataset_manifest,
            config={"permissions": permissions},
            version=version,
            overwrite=True,
        )
        print(f"Dataset '{name}' created with alias '{alias}' in gallery '{parent_id}'.")
    
    async def delete(self, artifact_id: str, delete_files: bool = True, recursive: bool = True) -> None:
        """Delete a dataset"""
        await self.ensure_connected()
        await self.connection.artifact_manager.delete(artifact_id=artifact_id, delete_files=delete_files, recursive=recursive)
        print(f"Dataset '{artifact_id}' deleted.")
    
    async def commit_dataset(self, alias: str, max_attempts: int = 5) -> bool:
        """Commit a dataset, with retries on failure"""
        commit_success = False
        commit_attempts = 0
        
        while not commit_success and commit_attempts < max_attempts:
            try:
                await self.ensure_connected()
                await asyncio.wait_for(
                    self.connection.artifact_manager.commit(alias),
                    timeout=Config.CONNECTION_TIMEOUT
                )
                print(f"Dataset '{alias}' committed successfully.")
                commit_success = True
            except Exception as e:
                commit_attempts += 1
                print(f"Error committing dataset (attempt {commit_attempts}/{max_attempts}): {str(e)}")
                await asyncio.sleep(5)
                await self.connection.reconnect()
        
        if not commit_success:
            print(f"WARNING: Failed to commit the dataset '{alias}' after {max_attempts} attempts.")
        
        return commit_success

async def create_gallery_example() -> None:
    """Example of creating a gallery and dataset"""
    gallery_manager = GalleryManager()
    
    try:
        # Create a gallery
        await gallery_manager.create_gallery(
            name="Image Map of U2OS FUCCI Drug Treatment",
            description="A collection for organizing imaging datasets acquired by microscopes",
            alias="agent-lens/image-map-u2os-fucci-drug-treatment",
            permissions = {"*": "r", "@": "*", "misty-teeth-42051243": "*","google-oauth2|103047988474094226050": "*"}
        )
    finally:
        await gallery_manager.connection.disconnect()

async def create_dataset_example() -> None:
    """Example of creating a dataset"""
    gallery_manager = GalleryManager()
    try:
        # Create a dataset in the gallery
        await gallery_manager.create_dataset(
            name="20250429-treatment",
            description="The Image Map of U2OS FUCCI Drug Treatment",
            alias="agent-lens/image-map-20250429-treatment",
            parent_id="agent-lens/image-map-u2os-fucci-drug-treatment",
            version="stage",
            permissions = {"*": "r", "@": "*", "misty-teeth-42051243": "*","google-oauth2|103047988474094226050": "*"}
        )
    finally:
        await gallery_manager.connection.disconnect()

    

async def delete_dataset_example() -> None:
    """Example of deleting a dataset"""
    gallery_manager = GalleryManager()
    await gallery_manager.connection.connect(client_id=None)
    await gallery_manager.delete(artifact_id="agent-lens/image-map-u2os-fucci-drug-treatment", delete_files=True, recursive=True)
    print("Dataset deleted.")
    await gallery_manager.connection.disconnect()

if __name__ == "__main__":
    asyncio.run(create_gallery_example()) 
    asyncio.run(create_dataset_example())