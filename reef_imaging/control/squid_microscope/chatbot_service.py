import logging

from tools.hypha_storage import HyphaDataStore
from tools.chatbot.aask import aask
import argparse
import asyncio

import numpy as np
from imjoy_rpc.hypha import login, connect_to_server, register_rtc_service

from squid_control.squid_controller import SquidController
login_required=True
current_x, current_y = 0,0

from hypha_manager import HyphaManager
from distutils.util import strtobool


# Initialize chatpt vision
from openai import AsyncOpenAI
import base64
from pydantic import Field, BaseModel
from typing import Optional, List
import httpx
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt


datastore = HyphaDataStore()
def encode_image(image_path):
    """This function is for ChatGPT Vision. To encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

class ImageInfo(BaseModel):
    """This class is for ChatGPT Vision. To define the image information."""
    url: str = Field(..., description="The URL of the image.")
    title: Optional[str] = Field(None, description="The title of the image.")

async def inspect_tool(images: List[dict]=Field(..., description="A list of images to be inspected, each with a http url and title"), query: str=Field(..., description="user query about the image"),  context_description: str=Field(..., description="describe the context for the visual inspection task")) -> str:
    """Inspect an image using GPT-Vision."""
    image_infos = [ImageInfo(**image) for image in images]
    for image in image_infos:
        assert image.url.startswith("http"), "Image URL must start with http."
    
    response = await aask(image_infos, [context_description, query])
    return response


def get_schema(context=None):
    return {
        "move_by_distance": {
            "type": "object",
            "title": "move_by_distance",
            "description": "Move the stage by a specified distance in millimeters, the stage will move along the X, Y, and Z axes. You must retur all three numbers. You also must return 0 if you don't to move the stage along that axis. Notice: for new well plate imaging, move the Z axis to 2.79mm can reach the focus position. And the maximum value of Z axis is 4.5mm.",
            "properties": {
                "x": {"type": "number", "description": "Move the stage along X axis", "default":  0 },
                "y": {"type": "number", "description": "Move the stage along Y axis", "default":  0 },
                "z": {"type": "number", "description": "Move the stage along Z axis", "default":  0 },
            },
        },
        "move_to_position": {
            "type": "object",
            "title": "move_to_position",
            "description": "Move the stage to a specified position in millimeters, the stage will move to the specified X, Y, and Z coordinates. You must retur all three numbers. You also must return 0 if you don't to move the stage along that axis.",
            "properties": {
                "x": {"type": "number", "description": "Move the stage to the X coordinate", "default": None},
                "y": {"type": "number", "description": "Move the stage to the Y coordinate", "default": None},
                "z": {"type": "number", "description": "Move the stage to the Z coordinate", "default": 3.35},
            },
        },
        "home_stage": {
            "type": "object",
            "title": "home_stage",
            "description": "The stage will move to the home position and recalibrate, then move to scanning position:(20,20,2)",
            "properties": {},
        },
        "auto_focus": {
            "type": "object",
            "title": "auto_focus",
            "description": "Autofocus to get and move to the best focus position for the current sample.",
            "properties": {
                "N": {"type": "number", "description": "Default value:10. This parameter represents the number of discrete focus positions that the autofocus algorithm evaluates to determine the optimal focus."},
                "delta_Z": {"type": "number", "description": "Default value: 1.524. This parameter defines the step size in the Z-axis between each focus position checked by the autofocus routine, and the unit is in micrometers."},
            },
        },
        "snap_image": {
            "type": "object",
            "title": "snap_image",
            "description": "Snap an image and return is the URL of the image. Your must show the image to user",
            "properties": {
                "exposure": {"type": "number", "description": "Set the microscope camera's exposure time in milliseconds."},
                "channel": {"type": "number", "description": "Set light source. Default value is 0. The illumination source and number is: [Bright Field=0, Fluorescence 405 nm=11, Fluorescence 488 nm=12,  Fluorescence 638 nm=13, Fluorescence 561 nm=14, Fluorescence 730 nm=15]."},
                "intensity": {"type": "number", "description": "Set the intensity of the illumination source. The default value for bright field is 5, for fluorescence is 100."},
            },
            "required": ["exposure", "channel", "intensity"]
        },
        "inspect_tool": {
            "type": "object",
            "title": "inspect_tool",
            "description": "Inspect the provided images visually based on the context, make insightful comments and answer questions about the provided images.",
            "properties": {
                "images": {
                    "type": "list",
                    "description": "A list of images to be inspected, each with a http url and title",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "list", "description": "The list organized byt URL and title of the images."},
                            "title": {"type": "string", "description": "The title of the images."},
                        },
                        "required": ["url"]
                    }
                },
                "query": {"type": "string", "description": "user query about the image"},
                "context_description": {"type": "string", "description": "describe the context for the visual inspection task"}
            },
            "required": ["images", "query", "context_description"]
        },
        "move_to_loading_position": {   
            "type": "object",
            "title": "move_to_loading_position",
            "description": "When sample need to be loaded or unloaded, move the stage to the zero position so that the robotic arm can reach the sample.",
            "properties": {},
        },
        "navigate_to_well": {
            "type": "object",
            "title": "navigate_to_well",
            "description": "Navigate to the specified well position in the well plate.",
            "properties": {
                "row": {"type": "string", "description": "The letter represents row number of the well position. Like 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'...."},
                "col": {"type": "number", "description": "The column number of the well position."},
                "wellplate_type": {"type": "string", "description": "The type of the well plate. Default type is '24', can be '6', '12', '24', '96', '384'."},
            },
            "required": ["row", "col", "wellplate_type"]
        }
    }



def move_to_position_schema(config, context=None):
    print("Moving the stage to position:", config)
    if config["x"] is None:
        config["x"] = 0
    if config["y"] is None:
        config["y"] = 0
    if config["z"] is None:
        config["z"] = 0
    result = hyphaManager.move_to_position(config["x"], config["y"], config["z"],context=context)
    return {"result": result}

def move_by_distance_schema(config, context=None):
    print("Moving the stage by distance:", config)
    if config["x"] is None:
        config["x"] = 0
    if config["y"] is None:
        config["y"] = 0
    if config["z"] is None:
        config["z"] = 0
    result = hyphaManager.move_by_distance(config["x"], config["y"], config["z"],context=context)
    return {"result": result}

def home_stage_schema(config, context=None):
    hyphaManager.home_stage(context=context)
    return {"result": "The stage is homed."}

def auto_focus_schema(config, context=None):
    hyphaManager.auto_focus(context=context)
    return {"result": "Auto focused!"}

def snap_image_schema(config, datastore=datastore,context=None):
    if config["exposure"] is None:
        config["exposure"] = 100
    if config["channel"] is None:
        config["channel"] = 0
    if config["intensity"] is None:
        config["intensity"] = 15
    squid_image_url = hyphaManager.snap(config["exposure"], config["channel"], config["intensity"],datastore=datastore, context=context)
    resp = f"![Image]({squid_image_url})"
    return resp

def move_to_loading_position_schema(config, context=None):
    hyphaManager.move_to_loading_position(context=context)
    return {"result": "Moved the stage to loading position!"}

def navigate_to_well_schema(config, context=None):
    hyphaManager.navigate_to_well(config["row"], config["col"], config["wellplate_type"],context=context)
    return {"result": "Moved the stage to the specified well position!"}

async def inspect_tool_schema(config, context=None):
    print("Inspecting the images with the context:", config)

    response = await inspect_tool(config["images"], config["query"], config["context_description"])
    return {"result": response}

async def setup():
    
    chatbot_extension = {
        "_rintf": True,
        "id": "squid-microscope-chatbot-extension",
        "type": "bioimageio-chatbot-extension",
        "name": "Squid Microscope Control",
        "description": "Your role: A chatbot controlling a microscope; Your mission: Answering the user's questions, and executing the commands to control the microscope; Definition of microscope: OBJECTIVES: 20x 'NA':0.4, You have one main camera and one autofocus camera. ",
        "config": {"visibility": "public", "require_context": True},
        "ping" : hyphaManager.ping,
        "get_schema": get_schema,
        "tools": {
            "move_by_distance": move_by_distance_schema,#@TODO: just move the stage. Can be switched to by distance or to position(is_relative=False/T)
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
        print("Failed to initialize squid controller\n", e,"\nTry to run with simulation mode")
        squidController = SquidController(is_simulation=True)

        
    hyphaManager = HyphaManager(squidController)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    loop = asyncio.get_event_loop()



    loop.create_task(setup())
    loop.run_forever()
    

