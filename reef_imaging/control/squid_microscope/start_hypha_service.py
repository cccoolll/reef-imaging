import os 
import logging
import time
import argparse
import asyncio
import fractions
from functools import partial
import traceback
import numpy as np
from hypha_rpc import login, connect_to_server, register_rtc_service
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCConfiguration
from aiortc.contrib.media import MediaPlayer, MediaRelay, MediaStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender
from av import VideoFrame
import json
import cv2

import sys
# Get the absolute path to the directory where squid_control is located
squid_control_path = os.path.abspath('/squid-control')

# Add this directory to sys.path
if squid_control_path not in sys.path:
    sys.path.append(squid_control_path)

# Now you can import squid_control
from squid_control.squid_controller import SquidController
from pydantic import Field, BaseModel
from typing import List, Optional
from tools.hypha_storage import HyphaDataStore
from tools.chatbot.aask import aask

login_required=True
current_x, current_y = 0,0
current_illumination_channel=None
current_intensity=None

global squidController
#squidController= SquidController(is_simulation=args.simulation)

def load_authorized_emails(login_required=True):
    if login_required:
        authorized_users_path = os.environ.get("BIOIMAGEIO_AUTHORIZED_USERS_PATH")
        if authorized_users_path:
            assert os.path.exists(
                authorized_users_path
            ), f"The authorized users file is not found at {authorized_users_path}"
            with open(authorized_users_path, "r") as f:
                authorized_users = json.load(f)["users"]
            authorized_emails = [
                user["email"] for user in authorized_users if "email" in user
            ]
        else:
            authorized_emails = None
    else:
        authorized_emails = None
    return authorized_emails

authorized_emails = load_authorized_emails()
print(f"Authorized emails: {authorized_emails}")

def check_permission(user):
    if user['is_anonymous']:
        return False
    if authorized_emails is None or user["email"] in authorized_emails:
        return True
    else:
        return False

async def ping(context=None):
    if login_required and context and context.get("user"):
        assert check_permission(
            context.get("user")
        ), "You don't have permission to use the chatbot, please sign up and wait for approval"
    return "pong"

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self):
        super().__init__()  # don't forget this!
        self.count = 0

    async def recv(self):
        # frame = await self.track.recv()
        print("Start snapping an image")
        img = np.random.randint(0, 155, (150, 300, 3)).astype('uint8')
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = self.count # frame.pts
        self.count+=1
        new_frame.time_base = fractions.Fraction(1, 1000)
        return new_frame



async def send_status(data_channel, workspace=None, token=None):
    """
    Send the current status of the microscope to the client. User can dump information of the microscope to a json data.
    ----------------------------------------------------------------
    Parameters
    ----------
    data_channel : aiortc.DataChannel
        The data channel to send the status to.
    workspace : str, optional
        The workspace to use. The default is None.
    token : str, optional
        The token to use. The default is None.

    Returns
    -------
    None.
    """
    while True:
        if data_channel and data_channel.readyState == "open":
            global current_x, current_y
            current_x, current_y, current_z, current_theta, is_illumination, _ = get_status()
            squid_status = {"x": current_x, "y": current_y, "z": current_z, "theta": current_theta, "illumination": is_illumination}
            data_channel.send(json.dumps(squid_status))
        await asyncio.sleep(1)  # Wait for 1 second before sending the next update


def move_by_distance(x,y,z, context=None):
    """
    Move the stage by a distance in x,y,z axis.
    ----------------------------------------------------------------
    Parameters
    ----------
    x : float
        The distance to move in x axis.
    y : float
        The distance to move in y axis.
    z : float
        The distance to move in z axis.
    context : dict, optional
            The context is a dictionary contains the following keys:
                - login_url: the login URL
                - report_url: the report URL
                - key: the key for the login
    """
    if not check_permission(context.get("user")):
        return "You don't have permission to use the chatbot, please contact us and wait for approval"
    is_success, x_pos, y_pos,z_pos, x_des, y_des, z_des =squidController.move_by_distance_limited(x,y,z)
    if is_success:
        result = f'The stage moved ({x},{y},{z})mm through x,y,z axis, from ({x_pos},{y_pos},{z_pos})mm to ({x_des},{y_des},{z_des})mm'
        print(result)
        return(result)
    else:
        result = f'The stage can not move ({x},{y},{z})mm through x,y,z axis, from ({x_pos},{y_pos},{z_pos})mm to ({x_des},{y_des},{z_des})mm because out of the range.'
        print(result)
        return(result)
        
def move_to_position(x,y,z, context=None):
    """
    Move the stage to a position in x,y,z axis.
    ----------------------------------------------------------------
    Parameters
    ----------
    x : float
        The distance to move in x axis.
    y : float
        The distance to move in y axis.
    z : float
        The distance to move in z axis.
    context : dict, optional
            The context is a dictionary contains keys:
                - login_url: the login URL
                - report_url: the report URL
                - key: the key for the login
            For detailes, see: https://ha.amun.ai/#/

    """
    if not check_permission(context.get("user")):
        return "You don't have permission to use the chatbot, please contact us and wait for approval"
    if x != 0:
        is_success, x_pos, y_pos,z_pos, x_des = squidController.move_x_to_limited(x)
        if not is_success:
            result = f'The stage can not move to position ({x},{y},{z})mm from ({x_pos},{y_pos},{z_pos})mm because out of the limit of X axis.'
            print(result)
            return(result)
            
    if y != 0:        
        is_success, x_pos, y_pos, z_pos, y_des = squidController.move_y_to_limited(y)
        if not is_success:
            result = f'X axis moved successfully, the stage is now at ({x_pos},{y_pos},{z_pos})mm. But aimed position is out of the limit of Y axis and the stage can not move to position ({x},{y},{z})mm.'
            print(result)
            return(result)
            
    if z != 0:    
        is_success, x_pos, y_pos, z_pos, z_des = squidController.move_z_to_limited(z)
        if not is_success:
            result = f'X and Y axis moved successfully, the stage is now at ({x_pos},{y_pos},{z_pos})mm. But aimed position is out of the limit of Z axis and stage can not move to position ({x},{y},{z})mm.'
            print(result)
            return(result)
            
    result = f'The stage moved to position ({x},{y},{z})mm from ({x_pos},{y_pos},{z_pos})mm successfully.'
    print(result)
    return(result)

def get_status(context=None):
    """
    Get the current status of the microscope.
    ----------------------------------------------------------------
    Parameters
    ----------
        context : dict, optional
            The context is a dictionary contains keys:
                - login_url: the login URL
                - report_url: the report URL
                - key: the key for the login
            For detailes, see: https://ha.amun.ai/#/

    Returns
    -------
    current_x : float
        The current position of the stage in x axis.
    current_y : float
        The current position of the stage in y axis.
    current_z : float
        The current position of the stage in z axis.
    current_theta : float
        The current position of the stage in theta axis.
    is_illumination_on : bool
        The status of the bright field illumination.

    """
    current_x, current_y, current_z, current_theta = squidController.navigationController.update_pos(microcontroller=squidController.microcontroller)
    is_illumination_on = squidController.liveController.illumination_on
    scan_channel = squidController.multipointController.selected_configurations
    return current_x, current_y, current_z, current_theta, is_illumination_on,scan_channel

    
def one_new_frame(context=None):
    print("Start snapping an image")
    gray_img=squidController.snap_image(0,50,100)
    print('The image is snapped')

    min_val = np.min(gray_img)
    max_val = np.max(gray_img)
    if max_val > min_val:  # Avoid division by zero if the image is completely uniform
        gray_img = (gray_img - min_val) * (255 / (max_val - min_val))
        gray_img = gray_img.astype(np.uint8)  # Convert to 8-bit image
    else:
        gray_img = np.zeros((512, 512), dtype=np.uint8)  # If no variation, return a black image
    bgr_img = np.stack((gray_img,)*3, axis=-1)  # Duplicate grayscale data across 3 channels to simulate BGR format.
    return bgr_img


def snap(exposure_time, channel, intensity, context=None):
    """ 
    Get the current frame from the camera, converted to a 3-channel BGR image.
    """
    if not check_permission(context.get("user")):
        return "You don't have permission to use the chatbot, please contact us and wait for approval"
    gray_img=squidController.snap_image(channel,intensity,exposure_time)
    # Rescale the image to span the full 0-255 range
    min_val = np.min(gray_img)
    max_val = np.max(gray_img)
    if max_val > min_val:  # Avoid division by zero if the image is completely uniform
        gray_img = (gray_img - min_val) * (255 / (max_val - min_val))
        gray_img = gray_img.astype(np.uint8)  # Convert to 8-bit image
    else:
        gray_img = np.zeros((512, 512), dtype=np.uint8)  # If no variation, return a black image
    # Resize the image to a standard size
    resized_img = cv2.resize(gray_img, (1006,759))
    bgr_img = np.stack((resized_img,)*3, axis=-1)  # Duplicate grayscale data across 3 channels to simulate BGR format.
    _, png_image = cv2.imencode('.png', bgr_img)
    # Store the PNG image
    file_id = datastore.put('file', png_image.tobytes(), 'snapshot.png', "Captured microscope image in PNG format")
    data_url= datastore.get_url(file_id)
    print(f'The image is snapped and saved as {data_url}')
    return data_url


def open_illumination(context=None):
    """
    Turn on the bright field illumination.
    ----------------------------------------------------------------
    Parameters
    ----------
    context : dict, optional
        The context is a dictionary contains keys:
            - login_url: the login URL
            - report_url: the report URL
            - key: the key for the login
        For detailes, see: https://ha.amun.ai/#/
    """
    if not check_permission(context.get("user")):
        return "You don't have permission to use the chatbot, please contact us and wait for approval"
    squidController.liveController.turn_on_illumination()

def close_illumination(context=None):
    """
    Turn off the bright field illumination.
    ----------------------------------------------------------------
    Parameters
    ----------
    context : dict, optional
        The context is a dictionary contains keys:
            - login_url: the login URL
            - report_url: the report URL
            - key: the key for the login
        For detailes, see: https://ha.amun.ai/#/
    """
    if not check_permission(context.get("user")):
        return "You don't have permission to use the chatbot, please contact us and wait for approval"
    squidController.liveController.turn_off_illumination()

def scan_well_plate(context=None):
    """
    Scan the well plate accroding to pre-defined position list.
    ----------------------------------------------------------------
    Parameters
    ----------
    context : dict, optional
        The context is a dictionary contains keys:
            - login_url: the login URL
            - report_url: the report URL
            - key: the key for the login
        For detailes, see: https://ha.amun.ai/#/
    """
    if not check_permission(context.get("user")):
        return "You don't have permission to use the chatbot, please contact us and wait for approval"
    print("Start scanning well plate")
    squidController.scan_well_plate(action_ID='Test')

def set_illumination(channel,intensity, context=None):
    """
    Set the intensity of the bright field illumination.
    illumination_source : int
    intensity : float, 0-100
    If you want to know the illumination source's and intensity's number, you can check the 'squid_control/channel_configurations.xml' file.
    """
    if not check_permission(context.get("user")):
        return "You don't have permission to use the chatbot, please contact us and wait for approval"
    squidController.liveController.set_illumination(channel,intensity)
    print(f'The intensity of the {channel} illumination is set to {intensity}.')

def set_camera_exposure(exposure_time, context=None):
    """
    Set the exposure time of the camera.
    exposure_time : float
    """
    if not check_permission(context.get("user")):
        return "You don't have permission to use the chatbot, please contact us and wait for approval"
    squidController.camera.set_exposure_time(exposure_time)
    print(f'The exposure time of the camera is set to {exposure_time}.')

def stop_scan(context=None):
    """
    Stop the well plate scanning.
    ----------------------------------------------------------------
    Parameters
    ----------
    context : dict, optional
        The context is a dictionary contains keys:
            - login_url: the login URL
            - report_url: the report URL
            - key: the key for the login
        For detailes, see: https://ha.amun.ai/#/
    """
    squidController.liveController.stop_live()
    print("Stop scanning well plate")
    pass

def home_stage(context=None):
    """
    Home the stage in z, y, and x axis.
    """
    if not check_permission(context.get("user")):
        return "You don't have permission to use the chatbot, please contact us and wait for approval"
    squidController.home_stage()
    print('The stage moved to home position in z, y, and x axis')


def move_to_loading_position(context=None):
    """
    Move the stage to the loading position.

    """
    if not check_permission(context.get("user")):
        return "You don't have permission to use the chatbot, please contact us and wait for approval"
    squidController.slidePositionController.move_to_slide_loading_position()
    print('The stage moved to loading position')

def auto_focus(context=None):
    """
    Auto focus the camera.

    """
    if not check_permission(context.get("user")):
        return "You don't have permission to use the chatbot, please contact us and wait for approval"
    squidController.do_autofocus()
    print('The camera is auto focused')

def navigate_to_well(row,col, wellplate_type, context=None):
    """
    Navigate to the specified well position in the well plate.
    row : int
    col : int
    wellplate_type : str, can be '6', '12', '24', '96', '384'
    """
    if not check_permission(context.get("user")):
        return "You don't have permission to use the chatbot, please contact us and wait for approval"
    if wellplate_type is None:
        wellplate_type = '96'
    squidController.platereader_move_to_well(row,col,wellplate_type)
    print(f'The stage moved to well position ({row},{col})')


#chatbot extension
class MoveByDistanceInput(BaseModel):
    x: float = Field(0, description="Move the stage along X axis")
    y: float = Field(0, description="Move the stage along Y axis")
    z: float = Field(0, description="Move the stage along Z axis")

class MoveToPositionInput(BaseModel):
    x: Optional[float] = Field(None, description="Move the stage to the X coordinate")
    y: Optional[float] = Field(None, description="Move the stage to the Y coordinate")
    z: float = Field(3.35, description="Move the stage to the Z coordinate")

class AutoFocusInput(BaseModel):
    N: int = Field(10, description="Number of discrete focus positions")
    delta_Z: float = Field(1.524, description="Step size in the Z-axis in micrometers")

class SnapImageInput(BaseModel):
    exposure: int = Field(..., description="Exposure time in milliseconds")
    channel: int = Field(..., description="Light source (e.g., 0 for Bright Field, 11 for Fluorescence 405 nm)")
    intensity: int = Field(..., description="Intensity of the illumination source")

class InspectToolInput(BaseModel):
    images: List[dict] = Field(..., description="A list of images to be inspected, each with a http url and title")
    query: str = Field(..., description="User query about the image")
    context_description: str = Field(..., description="Context for the visual inspection task")

class NavigateToWellInput(BaseModel):
    row: str = Field(..., description="Row number of the well position (e.g., 'A')")
    col: int = Field(..., description="Column number of the well position")
    wellplate_type: str = Field('24', description="Type of the well plate (e.g., '6', '12', '24', '96', '384')")

# Add chatbot-related functions
async def inspect_tool(images: List[dict], query: str, context_description: str) -> str:
    image_infos = [ImageInfo(**image) for image in images]
    for image in image_infos:
        assert image.url.startswith("http"), "Image URL must start with http."
    
    response = await aask(image_infos, [context_description, query])
    return response

# Modify existing functions to use Pydantic models
def move_by_distance_schema(config: MoveByDistanceInput, context=None):
    return move_by_distance(config.x, config.y, config.z, context)

def move_to_position_schema(config: MoveToPositionInput, context=None):
    return move_to_position(config.x or 0, config.y or 0, config.z or 0, context)

def auto_focus_schema(config: AutoFocusInput, context=None):
    return auto_focus(context)

def snap_image_schema(config: SnapImageInput, context=None):
    return snap(config.exposure, config.channel, config.intensity, context)

def navigate_to_well_schema(config: NavigateToWellInput, context=None):
    return navigate_to_well(config.row, config.col, config.wellplate_type, context)

async def inspect_tool_schema(config: InspectToolInput, context=None):
    response = await inspect_tool(config.images, config.query, config.context_description)
    return {"result": response}


datastore = HyphaDataStore()

# Add get_schema function
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


async def on_init(peer_connection):
    @peer_connection.on("track")
    def on_track(track):
        print(f"Track {track.kind} received")
        peer_connection.addTrack(
            VideoTransformTrack()
        )
        @track.on("ended")
        async def on_ended():
            print(f"Track {track.kind} ended")

    # data_channel = peer_connection.createDataChannel("microscopeStatus")
    # # Start the task to send stage position periodically
    # asyncio.create_task(send_status(data_channel))
    
async def start_hypha_service(server, service_id):



    
    await server.register_service(
        {
            "name": "Microscope Control Service",
            "id": "microscope-control-squid-2",
            "config":{
                "visibility": "public",
                #"run_in_executor": True
            },
            #"type": "echo",
            "move_by_distance": move_by_distance,
            "snap": snap,
            "off_illumination": close_illumination,
            "on_illumination": open_illumination,
            "set_illumination": set_illumination,
            "set_camera_exposure": set_camera_exposure,
            "scan_well_plate": scan_well_plate,
            "stop_scan": stop_scan,
            "home_stage": home_stage,
            "move_to_position": move_to_position,      
            "move_to_loading_position": move_to_loading_position,
            "auto_focus": auto_focus
        },
        overwrite=True
    )
    
    await register_rtc_service(
        server,
        service_id=f"{service_id}",
        config={
            "visibility": "public",
            "on_init": on_init,
        },
    )
    print(
        f"Service (service_id={service_id}) started successfully, available at https://hypha.aicell.io/{server.config.workspace}/services"
    )
    print(f"You can access the webrtc stream at https://cccoolll.github.io/reef-imaging/?service_id={service_id}")
    
async def start_chatbot_service(server, service_id):    
    # try:
    #     await datastore.setup(server, service_id="data-store")
    # except TypeError as e:
    #     if "Future" in str(e):
    #         # If config is a Future, wait for it to resolve
    #         config = await asyncio.wrap_future(server.config)
    #         await datastore.setup(server, service_id="data-store", config=config)
    #     else:
    #         raise e
    
    chatbot_extension = {
        "_rintf": True,
        "id": "squid-microscope-chatbot-extension",
        "type": "bioimageio-chatbot-extension",
        "name": "Squid Microscope Control",
        "description": "Your role: A chatbot controlling a microscope; Your mission: Answering the user's questions, and executing the commands to control the microscope; Definition of microscope: OBJECTIVES: 20x 'NA':0.4, You have one main camera and one autofocus camera.",
        "config": {"visibility": "public", "require_context": True},
        "ping": ping,
        "get_schema": get_schema,
        "tools": {
            "move_by_distance": move_by_distance_schema,
            "move_to_position": move_to_position_schema,
            "auto_focus": auto_focus_schema,
            "snap_image": snap_image_schema,
            "home_stage": home_stage,
            "move_to_loading_position": move_to_loading_position,
            "navigate_to_well": navigate_to_well_schema,
            "inspect_tool": inspect_tool_schema,
        }
    }

    await server.register_service(chatbot_extension, overwrite=True)
    

    print(f"You can access the chatbot at https://bioimage.io/chat?server=https://hypha.aicell.io&extension={chatbot_extension['id']}&assistant=Skyler")




async def setup(simulation=True):
    
    hypha_server_url = "https://hypha.aicell.io"
    hypha_server = await connect_to_server({"server_url": hypha_server_url})
    rtc_service_id = "squid-control-rtc"# if simulation else "squid-control-service"
    await start_hypha_service(hypha_server, rtc_service_id)
    
    # chatbot_id = "squid-microscope-chatbot-extension"
    # chatbot_server_url = "https://chat.bioimage.io"
    # token = await login({"server_url": chatbot_server_url})
    # chatbot_server = await connect_to_server({"server_url": chatbot_server_url, "token": token})
    # await start_chatbot_service(chatbot_server, chatbot_id)






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

    squidController = SquidController(is_simulation=args.simulation)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    loop = asyncio.get_event_loop()

    async def main():
        try:
            await setup(simulation=args.simulation)
        except Exception:
            traceback.print_exc()

    loop.create_task(main())
    loop.run_forever()
    
