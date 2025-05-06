
from hypha_rpc import connect_to_server
from io import BytesIO
from zipfile import ZipFile
import httpx

SERVER_URL = "https://hypha.aicell.io"


async def test_get_zip_file_content_endpoint(
    minio_server, fastapi_server, test_user_token
):
    """Test retrieving specific content and listing directories from a ZIP file stored in S3."""

    # Connect and get the artifact manager service
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
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

    # Create a ZIP file in memory
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("example.txt", "file contents of example.txt")
        zip_file.writestr("nested/example2.txt", "file contents of nested/example2.txt")
        zip_file.writestr(
            "nested/subdir/example3.txt", "file contents of nested/subdir/example3.txt"
        )
    zip_buffer.seek(0)

    # Upload the ZIP file to the artifact
    zip_file_path = "test-files"
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id,
        file_path=f"{zip_file_path}.zip",
        download_weight=1,
    )
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(put_url, data=zip_buffer.read())
        assert response.status_code == 200

    # Commit the dataset artifact
    await artifact_manager.commit(artifact_id=dataset.id)

    # Test retrieving `example.txt` from the ZIP file
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=example.txt"
        )
        assert response.status_code == 200
        assert response.text == "file contents of example.txt"

    # Test retrieving `nested/example2.txt` from the ZIP file
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=nested/example2.txt"
        )
        assert response.status_code == 200
        assert response.text == "file contents of nested/example2.txt"

    # Test retrieving a non-existent file
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=nonexistent.txt"
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "File not found inside ZIP: nonexistent.txt"

    # Test listing a non-existent directory
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=nonexistent/"
        )
        assert response.status_code == 200
        assert (
            response.json() == []
        )  # An empty list indicates an empty or non-existent directory.


