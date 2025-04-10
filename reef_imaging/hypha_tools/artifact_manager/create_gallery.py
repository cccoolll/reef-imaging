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

    # Create a collection for the Dataset Gallery
    gallery_manifest = {
        "name": "U2OS FUCCI Drug Treatment",
        "description": "A collection for organizing imaging datasets acquired by microscopes",
    }
    await artifact_manager.create(
        type="collection",
        alias="reef-imaging/u2os-fucci-drug-treatment",
        manifest=gallery_manifest,
        config={"permissions": {"*": "r+", "@": "r+"}},
        overwrite=True,
    )
    print("Dataset Gallery created.")


asyncio.run(main())
