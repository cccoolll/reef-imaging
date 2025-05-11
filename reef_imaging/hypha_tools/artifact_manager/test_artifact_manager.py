from core import connect_to_server
import httpx
from io import BytesIO
from zipfile import ZipFile
import asyncio
import os
import tempfile
import json
import numpy as np
SERVER_URL = "https://hypha.aicell.io"



async def test_get_zip_file_content_endpoint(
    test_user_token
):
    """Test retrieving specific content and listing directories from a ZIP file stored in S3."""

    # Connect and get the artifact manager service
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token, "workspace": "agent-lens"}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection and retrieve the UUID
    collection_manifest = {
        "name": "test-collection",
        "description": "A test collection",
    }
    collection = await artifact_manager.create(
        type="collection",
        manifest=collection_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Create a dataset within the collection
    dataset_manifest = {
        "name": "test-dataset",
        "description": "A test dataset",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest=dataset_manifest,
        version="stage",
    )

    # Create a ZIP file on disk
    temp_dir = tempfile.gettempdir()
    zip_file_path_on_disk = os.path.join(temp_dir, "test-files.zip")
    
    with ZipFile(zip_file_path_on_disk, "w") as zip_file:
        # Add regular files
        zip_file.writestr("example.txt", "file contents of example.txt")
        zip_file.writestr("nested/example2.txt", "file contents of nested/example2.txt")
        zip_file.writestr(
            "nested/subdir/example3.txt", "file contents of nested/subdir/example3.txt"
        )
        
        # Create and add .zarr chunk files
        # Create zarr metadata files
        zarr_attrs = {"_ARRAY_DIMENSIONS": ["z", "y", "x"], "datatype": "float32"}
        zarr_attrs_json = json.dumps(zarr_attrs)
        zip_file.writestr("nested-zarr/.zattrs", zarr_attrs_json)
        
        # Create 1100 chunks of 1x1 to verify the file size's affect on the artifact manager
        chunk_shape = [246, 256]
        array_shape = [11, 10, 10]  # Shape of the entire array

        
        zarr_array = {
            "chunks": chunk_shape,
            "compressor": {"id": "zlib", "level": 1},
            "dtype": "<f4",
            "fill_value": "NaN",
            "filters": None,
            "order": "C",
            "shape": [chunk_shape[0] * array_shape[0], chunk_shape[1] * array_shape[1], array_shape[2]],
            "zarr_format": 2
        }
        zarr_array_json = json.dumps(zarr_array)
        zip_file.writestr("nested-zarr/.zarray", zarr_array_json)
        
        # Create some chunk files with random data
        # We'll create a 3D array (z, y, x) with chunks across all dimensions
        print("Creating Zarr chunks for approximately 100MB file size...")
        total_chunks = array_shape[0] * array_shape[1] * array_shape[2]
        chunk_count = 0
        
        # Create a reusable chunk of random data to save memory
        chunk_data = np.random.rand(chunk_shape[0], chunk_shape[1]).astype(np.float32)
        chunk_bytes = chunk_data.tobytes()
        
        # Add chunks to the zip file
        for z in range(array_shape[2]):  # 10
            for y in range(array_shape[1]):  # 20
                for x in range(array_shape[0]):  # 20
                    # Add to zip file with zarr chunk naming convention
                    zip_file.writestr(f"nested-zarr/{x}.{y}.{z}", chunk_bytes)
                    
                    # Print progress
                    chunk_count += 1
                    if chunk_count % 100 == 0:
                        progress = (chunk_count / total_chunks) * 100
                        print(f"Progress: {progress:.1f}% ({chunk_count}/{total_chunks} chunks)")
        
        # Add a zarr group
        zarr_group = {"zarr_format": 2}
        zarr_group_json = json.dumps(zarr_group)
        zip_file.writestr("nested-zarr/subgroup/.zgroup", zarr_group_json)
        
        # Add a README in the zarr directory
        zip_file.writestr("nested-zarr/README.md", "This is a zarr directory containing a large 3D array with 256x256 chunks")
    
    # Get the size of the ZIP file
    zip_size_mb = os.path.getsize(zip_file_path_on_disk) / (1024 * 1024)
    print(f"Created ZIP file: {zip_file_path_on_disk} (Size: {zip_size_mb:.2f} MB)")
    
    # Upload the ZIP file to the artifact
    zip_file_path = "test-files"
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id,
        file_path=f"{zip_file_path}.zip",
        download_weight=1,
    )
    print(f"put_url: {put_url}, dataset.id: {dataset.id}")
    async with httpx.AsyncClient(timeout=300) as client:
        # Use data parameter with the file contents for async compatibility
        with open(zip_file_path_on_disk, "rb") as f:
            file_data = f.read()
        response = await client.put(put_url, data=file_data)
        assert response.status_code == 200

    # Commit the dataset artifact
    await artifact_manager.commit(artifact_id=dataset.id)
    print(f"you can test the dataset at {SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip")
    # Test retrieving `example.txt` from the ZIP file
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip"
        )
        assert response.status_code == 200

    # Test retrieving `nested/example2.txt` from the ZIP file
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=nested/example2.txt"
        )
        assert response.status_code == 200
        assert response.text == "file contents of nested/example2.txt"
    print(f"url: {SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=example.txt")
    # Test retrieving a non-existent file
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=nonexistent.txt"
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "File not found inside ZIP: nonexistent.txt"

    # Test listing a non-existent directory
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=nonexistent/"
        )
        assert response.status_code == 200
        assert (
            response.json() == []
        )  # An empty list indicates an empty or non-existent directory.
        
    # Test retrieving a zarr metadata file
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=nested-zarr/.zarray"
        )
        assert response.status_code == 200
        assert json.loads(response.text)["zarr_format"] == 2

    # Clean up the temporary ZIP file
    os.remove(zip_file_path_on_disk)


token=os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
asyncio.run(test_get_zip_file_content_endpoint(token))