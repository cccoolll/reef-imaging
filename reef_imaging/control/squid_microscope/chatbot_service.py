import logging
import argparse
import asyncio
import base64
from typing import List, Optional

from pydantic import Field, BaseModel
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from distutils.util import strtobool

from tools.hypha_storage import HyphaDataStore
from tools.chatbot.aask import aask
from imjoy_rpc.hypha import login, connect_to_server, register_rtc_service
from squid_control.squid_controller import SquidController
from hypha_manager import HyphaManager

# Initialize global variables
login_required = True
current_x, current_y = 0, 0
datastore = HyphaDataStore()

# Define Pydantic models for schemas
class MoveByDistanceInput(BaseModel):
    """Move the stage by a specified distance in millimeters."""
    x: float = Field(0, description="Move the stage along X axis")
    y: float = Field(0, description="Move the stage along Y axis")
    z: float = Field(0, description="Move the stage along Z axis")

class MoveToPositionInput(BaseModel):
    """Move the stage to a specified position in millimeters."""
    x: Optional[float] = Field(None, description="Move the stage to the X coordinate")
    y: Optional[float] = Field(None, description="Move the stage to the Y coordinate")
    z: float = Field(3.35, description="Move the stage to the Z coordinate")

class AutoFocusInput(BaseModel):
    """Autofocus to get and move to the best focus position for the current sample."""
    N: int = Field(10, description="Number of discrete focus positions")
    delta_Z: float = Field(1.524, description="Step size in the Z-axis in micrometers")

class SnapImageInput(BaseModel):
    """Snap an image and return the URL of the image."""
    exposure: int = Field(..., description="Exposure time in milliseconds")
    channel: int = Field(..., description="Light source (e.g., 0 for Bright Field, 11 for Fluorescence 405 nm)")
    intensity: int = Field(..., description="Intensity of the illumination source")

class InspectToolInput(BaseModel):
    """Inspect the provided images visually based on the context."""
    images: List[dict] = Field(..., description="A list of images to be inspected, each with a http url and title")
    query: str = Field(..., description="User query about the image")
    context_description: str = Field(..., description="Context for the visual inspection task")

class NavigateToWellInput(BaseModel):
    """Navigate to the specified well position in the well plate."""
    row: str = Field(..., description="Row number of the well position (e.g., 'A')")
    col: int = Field(..., description="Column number of the well position")
    wellplate_type: str = Field('24', description="Type of the well plate (e.g., '6', '12', '24', '96', '384')")

# Define functions to handle the operations
def encode_image(image_path):
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

def move_to_position_schema(config: MoveToPositionInput, context=None):
    print("Moving the stage to position:", config)
    result = hyphaManager.move_to_position(config.x or 0, config.y or 0, config.z or 0, context=context)
    return {"result": result}

def move_by_distance_schema(config: MoveByDistanceInput, context=None):
    print("Moving the stage by distance:", config)
    result = hyphaManager.move_by_distance(config.x, config.y, config.z, context=context)
    return {"result": result}

def home_stage_schema(config, context=None):
    hyphaManager.home_stage(context=context)
    return {"result": "The stage is homed."}

def auto_focus_schema(config: AutoFocusInput, context=None):
    hyphaManager.auto_focus(N=config.N, delta_Z=config.delta_Z, context=context)
    return {"result": "Auto focused!"}

def snap_image_schema(config: SnapImageInput, datastore=datastore, context=None):
    squid_image_url = hyphaManager.snap(config.exposure, config.channel, config.intensity, datastore=datastore, context=context)
    resp = f"![Image]({squid_image_url})"
    return resp

def move_to_loading_position_schema(config, context=None):
    hyphaManager.move_to_loading_position(context=context)
    return {"result": "Moved the stage to loading position!"}

def navigate_to_well_schema(config: NavigateToWellInput, context=None):
    hyphaManager.navigate_to_well(config.row, config.col, config.wellplate_type, context=context)
    return {"result": "Moved the stage to the specified well position!"}

async def inspect_tool_schema(config: InspectToolInput, context=None):
    print("Inspecting the images with the context:", config)
    response = await inspect_tool(config.images, config.query, config.context_description)
    return {"result": response}

async def inspect_tool(images: List[dict], query: str, context_description: str) -> str:
    """Inspect an image using GPT-Vision."""
    image_infos = [ImageInfo(**image) for image in images]
    for image in image_infos:
        assert image.url.startswith("http"), "Image URL must start with http."
    
    response = await aask(image_infos, [context_description, query])
    return response

# Define the schema getter function
def get_schema(context=None):
    return {
        "move_by_distance": MoveByDistanceInput.schema(),
        "move_to_position": MoveToPositionInput.schema(),
        "home_stage": {"type": "object", "properties": {}},
        "auto_focus": AutoFocusInput.schema(),
        "snap_image": SnapImageInput.schema(),
        "inspect_tool": InspectToolInput.schema(),
        "move_to_loading_position": {"type": "object", "properties": {}},
        "navigate_to_well": NavigateToWellInput.schema()
    }

# Async setup function
async def setup():
    chatbot_extension = {
        "_rintf": True,
        "id": "squid-microscope-chatbot-extension",
        "type": "bioimageio-chatbot-extension",
        "name": "Squid Microscope Control",
        "description": "Your role: A chatbot controlling a microscope; Your mission: Answering the user's questions, and executing the commands to control the microscope; Definition of microscope: OBJECTIVES: 20x 'NA':0.4, You have one main camera and one autofocus camera.",
        "config": {"visibility": "public", "require_context": True},
        "ping": hyphaManager.ping,
        "get_schema": get_schema,
        "tools": {
            "move_by_distance": move_by_distance_schema,
            "move_to_position": move_to_position_schema,
            "auto_focus": auto_focus_schema,
            "snap_image": snap_image_schema,
            "home_stage": home_stage_schema,
            "move_to_loading_position": move_to_loading_position_schema,
            "navigate_to_well": navigate_to_well_schema,
            "inspect_tool": inspect_tool_schema,
        }
    }

    server_url = "https://chat.bioimage.io"
    token = await login({"server_url": server_url})
    server = await connect_to_server({"server_url": server_url, "token": token})
    svc = await server.register_service(chatbot_extension)

    await datastore.setup(server, service_id="data-store")

    print(f"Extension service registered with id: {svc.id}, you can visit the service at:\n https://bioimage.io/chat?server={server_url}&extension={svc.id}&assistant=Skyler")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Squid microscope control services for Hypha."
    )
    parser.add_argument(
        "--simulation",
        dest="simulation",
        action="store_true",
        default=True,
        help="Run in simulation mode (default: True)"
    )
    parser.add_argument(
        "--no-simulation",
        dest="simulation",
        action="store_false",
        help="Run without simulation mode"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    try:
        squidController = SquidController(is_simulation=args.simulation)
    except Exception as e:
        print("Failed to initialize squid controller\n", e, "\nTry to run with simulation mode")
        squidController = SquidController(is_simulation=True)

    hyphaManager = HyphaManager(squidController)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    loop = asyncio.get_event_loop()
    loop.create_task(setup())
    loop.run_forever()
