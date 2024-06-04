import logging

from tools.hypha_storage import HyphaDataStore
import argparse
import asyncio

import numpy as np
from imjoy_rpc.hypha import login, connect_to_server, register_rtc_service

from squid_control.squid_controller import SquidController
login_required=True
current_x, current_y = 0,0

from hypha_manager import HyphaManager
from distutils.util import strtobool

# hyphaManager = HyphaManager()


# Now define chatbot services
global datastore
datastore = HyphaDataStore()

def get_schema(context=None):
    return {
        "move_by_distance": {
            "type": "bioimageio-chatbot-extension",
            "title": "move_by_distance",
            "description": "Move the stage by a specified distance in millimeters, the stage will move along the X, Y, and Z axes. You must retur all three numbers. You also must return 0 if you don't to move the stage along that axis. Notice: for new well plate imaging, move the Z axis to 4.1mm can reach the focus position. And the maximum value of Z axis is 5mm.",
            "properties": {
                "x": {"type": "number", "description": "Move the stage along X axis, default is 0."},
                "y": {"type": "number", "description": "Move the stage along Y axis, default is 0."},
                "z": {"type": "number", "description": "Move the stage along Z axis,default is 0."},
            },
        },
        "move_to_position": {
            "type": "bioimageio-chatbot-extension",
            "title": "move_to_position",
            "description": "Move the stage to a specified position in millimeters, the stage will move to the specified X, Y, and Z coordinates. You must retur all three numbers. You also must return 0 if you don't to move the stage along that axis.",
            "properties": {
                "x": {"type": "number", "description": "Move the stage to the X coordinate, default is 0."},
                "y": {"type": "number", "description": "Move the stage to the Y coordinate, default is 0."},
                "z": {"type": "number", "description": "Move the stage to the Z coordinate, default is 0."},
            },
        },
        "home_stage": {
            "type": "bioimageio-chatbot-extension",
            "title": "home_stage",
            "description": "The stage will move to the home position and recalibrate, then move to scanning position:(20,20,2)",
            "properties": {
                "is_home": {"type": "boolean", "description": "True if the stage is homed, False if the stage is not homed."},
            },
        },
        "auto_focus": {
            "type": "bioimageio-chatbot-extension",
            "title": "auto_focus",
            "description": "Autofocus the microscope, the value returned is just 1. If this action is required, it will execute before snapping an image.",
            "properties": {
                "N": {"type": "number", "description": "Default value:10. This parameter represents the number of discrete focus positions that the autofocus algorithm evaluates to determine the optimal focus."},
                "delta_Z": {"type": "number", "description": "Default value: 1.524. This parameter defines the step size in the Z-axis between each focus position checked by the autofocus routine, and the unit is in micrometers."},
            },
        },
        "snap_image": {
            "type": "bioimageio-chatbot-extension",
            "title": "snap_image",
            "description": "Snap an image and show it to user. The value returned is the URL of the image.",
            "properties": {
                "exposure": {"type": "number", "description": "Set the microscope camera's exposure time in milliseconds."},
                "channel": {"type": "number", "description": "Set light source. Default value is 0. The illumination source and number is: [Bright Field=0, Fluorescence 405 nm=11, Fluorescence 488 nm=12,  Fluorescence 638 nm=13, Fluorescence 561 nm=14, Fluorescence 730 nm=15]."},
                "intensity": {"type": "number", "description": "Set the intensity of the illumination source. The default value for bright field is 15, for fluorescence is 100."},
            },  
        },
        "move_to_loading_position": {   
            "type": "bioimageio-chatbot-extension",
            "title": "move_to_loading_position",
            "description": "When sample need to be loaded or unloaded, move the stage to the zero position so that the robotic arm can reach the sample.",
            "properties": {
                "is_loading": {"type": "boolean", "description": "True if the sample is being loaded, False if the sample is being unloaded."},
            },
        },
        "navigate_to_well": {
            "type": "bioimageio-chatbot-extension",
            "title": "navigate_to_well",
            "description": "Navigate to the specified well position in the well plate.",
            "properties": {
                "row": {"type": "string", "description": "The letter represents row number of the well position. Like 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'...."},
                "col": {"type": "number", "description": "The column number of the well position."},
                "wellplate_type": {"type": "string", "description": "The type of the well plate. Default type is '24', can be '6', '12', '24', '96', '384'."},
            },
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

async def setup():
    
    chatbot_extension = {
        "_rintf": True,
        "id": "squid-control",
        "type": "bioimageio-chatbot-extension",
        "name": "Squid Microscope Control",
        "description": "Your role: A chatbot controlling a microscope; Your mission: Answering the user's questions, and executing the commands to control the microscope; Definition of microscope: OBJECTIVES: 20x 'NA':0.4, You have one main camera and one autofocus camera. ",
        "config": {"visibility": "public", "require_context": True},
        "ping" : hyphaManager.ping,
        "get_schema": get_schema,
        "tools": {
            "move_by_distance": move_by_distance_schema,
            "move_to_position": move_to_position_schema, 
            "auto_focus": auto_focus_schema, 
            "snap_image": snap_image_schema,
            "home_stage": home_stage_schema,
            "move_to_loading_position": move_to_loading_position_schema,
            "navigate_to_well": navigate_to_well_schema,
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
    parser.add_argument("--simulation", type=lambda x: bool(strtobool(x)), default=False, help="The simulation mode")
    parser.add_argument("--service-id", type=str, default="squid-control", help="The service id")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    try:
        squidController = SquidController(is_simulation=args.simulation)
    except Exception as e:
        print(f"Failed to initialize Squid Controller.\n Eorror:\n{e}\n ========================================================\nUse simulation mode instead.")
        squidController = SquidController(is_simulation=True)
    hyphaManager = HyphaManager(squidController, args.service_id)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    loop = asyncio.get_event_loop()
    tasks = [
        loop.create_task(setup())
        ]

    # Register a callback for when the asyncio loop closes to handle any cleanup
    for task in tasks:
        task.add_done_callback(lambda t: loop.stop() if t.exception() else None)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("Shutting down gracefully")
    finally:
        # Gather all tasks and cancel them to ensure clean exit
        all_tasks = asyncio.all_tasks(loop)
        for t in all_tasks:
            t.cancel()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

    
