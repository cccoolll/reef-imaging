import os
import asyncio
from hypha_rpc import connect_to_server
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("REEF_WORKSPACE_TOKEN")
SERVER_URL = "https://hypha.aicell.io"


async def main():
    # Connect to the Artifact Manager API
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Add a dataset to the gallery
    dataset_manifest = {
        "name": "image-map-20250410-treatment-full",
        "description": "The Image Map of U2OS FUCCI Drug Treatment, full time lapse",
    }
    await artifact_manager.create(
        parent_id="reef-imaging/image-map-of-u2os-fucci-drug-treatment",
        alias="image-map-20250410-treatment-full",
        manifest=dataset_manifest,
        version="stage",
        overwrite=True,
    )
    print("Dataset added to the gallery.")


asyncio.run(main())
