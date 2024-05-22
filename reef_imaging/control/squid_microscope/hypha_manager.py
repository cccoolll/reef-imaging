import os 
import logging


import pyqtgraph.dockarea as dock
import time
from tools.hypha_storage import HyphaDataStore
import argparse
import asyncio
import fractions

import numpy as np
#from av import VideoFrame
from imjoy_rpc.hypha import login, connect_to_server, register_rtc_service
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCConfiguration

from aiortc.contrib.media import MediaPlayer, MediaRelay, MediaStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender
from av import VideoFrame
import fractions
import json
import webbrowser
from squid_control.squid_controller import SquidController
#import squid_control.squid_chatbot as chatbot
import cv2

current_x, current_y = 0,0
#squidController= SquidController(is_simulation=args.simulation)

class HyphaManager:
    def __init__(self, squidController,login_required=True):
        self.squidController = squidController
        self.datastore = HyphaDataStore()
        self.squidController = squidController
        self.login_required = login_required
        self.authorized_emails = None
        print(f"Authorized emails: {self.authorized_emails}")

    def load_authorized_emails(self):
        if self.login_required:
            authorized_users_path = os.environ.get("BIOIMAGEIO_AUTHORIZED_USERS_PATH")
            if authorized_users_path:
                assert os.path.exists(
                    authorized_users_path
                ), f"The authorized users file is not found at {authorized_users_path}"
                with open(authorized_users_path, "r") as f:
                    authorized_users = json.load(f)["users"]
                self.authorized_emails = [
                    user["email"] for user in authorized_users if "email" in user
                ]
            else:
                self.authorized_emails = None
        else:
            self.authorized_emails = None



    def check_permission(self, user):
        if user['is_anonymous']:
            return False
        if self.authorized_emails is None or user["email"] in self.authorized_emails:
            return True
        else:
            return False

    async def ping(self,context=None):
        if self.login_required and context and context.get("user"):
            assert self.check_permission(
                context.get("user")
            ), "You don't have permission to use the chatbot, please sign up and wait for approval"
        return "pong"

    class VideoTransformTrack(MediaStreamTrack):
        """
        A video stream track that transforms frames from another track.
        """

        kind = "video"

        def __init__(self):
            super().__init__()  # don't forget this!
            self.count = 0

        async def recv(self):
            # Read frame from squid controller, now correctly formatted as BGR
            bgr_img = self.one_new_frame()
            # Create the video frame
            new_frame = VideoFrame.from_ndarray(bgr_img, format="bgr24")
            new_frame.pts = self.count
            new_frame.time_base = fractions.Fraction(1, 1000)
            self.count += 1
            await asyncio.sleep(1)  # Simulating frame rate delay
            return new_frame



    async def send_status(self,data_channel, workspace=None, token=None):
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
                current_x, current_y, current_z, current_theta, is_illumination, _ = self.get_status()
                squid_status = {"x": current_x, "y": current_y, "z": current_z, "theta": current_theta, "illumination": is_illumination}
                data_channel.send(json.dumps(squid_status))
            await asyncio.sleep(1)  # Wait for 1 second before sending the next update


    def move_by_distance(self,x,y,z, context=None):
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
        if not self.check_permission(context.get("user")):
            return "You don't have permission to use the chatbot, please contact us and wait for approval"
        is_success, x_pos, y_pos,z_pos, x_des, y_des, z_des =self.squidController.move_by_distance_safely(x,y,z)
        if is_success:
            result = f'The stage moved ({x},{y},{z})mm through x,y,z axis, from ({x_pos},{y_pos},{z_pos})mm to ({x_des},{y_des},{z_des})mm'
            print(result)
            return(result)
        else:
            result = f'The stage can not move ({x},{y},{z})mm through x,y,z axis, from ({x_pos},{y_pos},{z_pos})mm to ({x_des},{y_des},{z_des})mm because out of the range.'
            print(result)
            return(result)
            
    def move_to_position(self, x,y,z, context=None):
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
        if not self.check_permission(context.get("user")):
            return "You don't have permission to use the chatbot, please contact us and wait for approval"
        if x != 0:
            is_success, x_pos, y_pos,z_pos, x_des = self.squidController.move_x_to_safely(x)
            if not is_success:
                result = f'The stage can not move to position ({x},{y},{z})mm from ({x_pos},{y_pos},{z_pos})mm because out of the limit of X axis.'
                print(result)
                return(result)
                
        if y != 0:        
            is_success, x_pos, y_pos, z_pos, y_des = self.squidController.move_y_to_safely(y)
            if not is_success:
                result = f'X axis moved successfully, the stage is now at ({x_pos},{y_pos},{z_pos})mm. But aimed position is out of the limit of Y axis and the stage can not move to position ({x},{y},{z})mm.'
                print(result)
                return(result)
                
        if z != 0:    
            is_success, x_pos, y_pos, z_pos, z_des = self.squidController.move_z_to_safely(z)
            if not is_success:
                result = f'X and Y axis moved successfully, the stage is now at ({x_pos},{y_pos},{z_pos})mm. But aimed position is out of the limit of Z axis and stage can not move to position ({x},{y},{z})mm.'
                print(result)
                return(result)
                
        result = f'The stage moved to position ({x},{y},{z})mm from ({x_pos},{y_pos},{z_pos})mm successfully.'
        print(result)
        return(result)

    def get_status(self, context=None):
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
        current_x, current_y, current_z, current_theta = self.squidController.navigationController.update_pos(microcontroller=self.squidController.microcontroller)
        is_illumination_on = self.squidController.liveController.illumination_on
        scan_channel = self.squidController.multipointController.selected_configurations
        return current_x, current_y, current_z, current_theta, is_illumination_on,scan_channel


    def one_new_frame(self, context=None):
        gray_img = self.squidController.camera.read_frame()
        bgr_img = np.stack((gray_img,)*3, axis=-1)  # Duplicate grayscale data across 3 channels to simulate BGR format.
        return bgr_img


    def snap(self, exposure_time, channel, intensity,context=None):
        """
        Get the current frame from the camera, converted to a 3-channel BGR image.
        """
        if not self.check_permission(context.get("user")):
            return "You don't have permission to use the chatbot, please contact us and wait for approval"
        if exposure_time is None:
            exposure_time = 100
        if channel is None:
            channel = 0
        if intensity is None:
            intensity = 15
        self.squidController.camera.set_exposure_time(exposure_time)
        self.squidController.camera.send_trigger()
        self.squidController.liveController.turn_on_illumination()
        self.squidController.liveController.set_illumination(channel,intensity)
        if self.squidController.microcontroller.is_busy():
            time.sleep(0.05)
        gray_img = self.squidController.camera.read_frame()
        time.sleep(0.05)
        #squidController.liveController.set_illumination(0,0)
        if self.squidController.microcontroller.is_busy():
            time.sleep(0.005)
        self.squidController.liveController.turn_off_illumination()
        #gray_img=np.resize(gray_img,(512,512))
        # Rescale the image to span the full 0-255 range
        min_val = np.min(gray_img)
        max_val = np.max(gray_img)
        if max_val > min_val:  # Avoid division by zero if the image is completely uniform
            gray_img = (gray_img - min_val) * (255 / (max_val - min_val))
            gray_img = gray_img.astype(np.uint8)  # Convert to 8-bit image
        else:
            gray_img = np.zeros((512, 512), dtype=np.uint8)  # If no variation, return a black image

        bgr_img = np.stack((gray_img,)*3, axis=-1)  # Duplicate grayscale data across 3 channels to simulate BGR format.
        _, png_image = cv2.imencode('.png', bgr_img)
        # Store the PNG image
        file_id = self.datastore.put('file', png_image.tobytes(), 'snapshot.png', "Captured microscope image in PNG format")
        print(f'The image is snapped and saved as {self.datastore.get_url(file_id)}')
        return self.datastore.get_url(file_id)


    def open_illumination(self, context=None):
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
        if not self.check_permission(context.get("user")):
            return "You don't have permission to use the chatbot, please contact us and wait for approval"
        self.squidController.liveController.turn_on_illumination()

    def close_illumination(self, context=None):
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
        if not self.check_permission(context.get("user")):
            return "You don't have permission to use the chatbot, please contact us and wait for approval"
        self.squidController.liveController.turn_off_illumination()

    def scan_well_plate(self, context=None):
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
        if not self.check_permission(context.get("user")):
            return "You don't have permission to use the chatbot, please contact us and wait for approval"
        print("Start scanning well plate")
        self.squidController.scan_well_plate(action_ID='Test')

    def set_illumination(self, illumination_source,intensity, context=None):
        """
        Set the intensity of the bright field illumination.
        illumination_source : int
        intensity : float, 0-100
        If you want to know the illumination source's and intensity's number, you can check the 'squid_control/channel_configurations.xml' file.
        """
        if not self.check_permission(context.get("user")):
            return "You don't have permission to use the chatbot, please contact us and wait for approval"
        self.squidController.liveController.set_illumination(illumination_source,intensity)
        print(f'The intensity of the {illumination_source} illumination is set to {intensity}.')



    def stop_scan(self, context=None):
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
        self.squidController.liveController.stop_live()
        print("Stop scanning well plate")
        pass

    def home_stage(self, context=None):
        """
        Home the stage in z, y, and x axis.
        """
        if not self.check_permission(context.get("user")):
            return "You don't have permission to use the chatbot, please contact us and wait for approval"
        self.squidController.home_stage()
        print('The stage moved to home position in z, y, and x axis')


    def move_to_loading_position(self, context=None):
        """
        Move the stage to the loading position.

        """
        if not self.check_permission(context.get("user")):
            return "You don't have permission to use the chatbot, please contact us and wait for approval"
        self.squidController.slidePositionController.move_to_slide_loading_position()
        print('The stage moved to loading position')

    def auto_focus(self, context=None):
        """
        Auto focus the camera.

        """
        if not self.check_permission(context.get("user")):
            return "You don't have permission to use the chatbot, please contact us and wait for approval"
        self.squidController.do_autofocus()
        print('The camera is auto focused')

    def navigate_to_well(self, row,col, wellplate_type, context=None):
        """
        Navigate to the specified well position in the well plate.
        row : int
        col : int
        wellplate_type : str, can be '6', '12', '24', '96', '384'
        """
        if not self.check_permission(context.get("user")):
            return "You don't have permission to use the chatbot, please contact us and wait for approval"
        if wellplate_type is None:
            wellplate_type = '24'
        self.squidController.platereader_move_to_well(row,col,wellplate_type)
        print(f'The stage moved to well position ({row},{col})')



    
