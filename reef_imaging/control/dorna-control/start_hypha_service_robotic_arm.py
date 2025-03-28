import asyncio
import argparse
import os
from hypha_rpc import connect_to_server, login
from dorna2 import Dorna
import dotenv
from pydantic import Field
from hypha_rpc.utils.schema import schema_function

dotenv.load_dotenv()  
ENV_FILE = dotenv.find_dotenv()  
if ENV_FILE:  
    dotenv.load_dotenv(ENV_FILE)  

class RoboticArmService:
    def __init__(self, local):
        self.local = local
        self.server_url = "http://localhost:9527" if local else "https://hypha.aicell.io"
        self.robot = Dorna()
        self.ip = "192.168.2.20"
        self.connected = False
        # Add task status tracking
        self.task_status = {
            "move_sample_from_microscope1_to_incubator": "not_started",
            "move_sample_from_incubator_to_microscope1": "not_started",
            "grab_sample_from_microscope1": "not_started",
            "grab_sample_from_incubator": "not_started",
            "put_sample_on_microscope1": "not_started",
            "put_sample_on_incubator": "not_started",
            "transport_from_incubator_to_microscope1": "not_started",
            "transport_from_microscope1_to_incubator": "not_started",
            "connect": "not_started",
            "disconnect": "not_started",
            "halt": "not_started",
            "get_all_joints": "not_started",
            "get_all_positions": "not_started",
            "light_on": "not_started",
            "light_off": "not_started"
        }

    async def start_hypha_service(self, server):
        svc = await server.register_service({
            "name": "Robotic Arm Control",
            "id": "robotic-arm-control",
            "config": {
                "visibility": "public",
                "run_in_executor": True
            },
            "move_sample_from_microscope1_to_incubator": self.move_sample_from_microscope1_to_incubator,
            "move_sample_from_incubator_to_microscope1": self.move_sample_from_incubator_to_microscope1,
            "grab_sample_from_microscope1": self.grab_sample_from_microscope1,
            "grab_sample_from_incubator": self.grab_sample_from_incubator,
            "put_sample_on_microscope1": self.put_sample_on_microscope1,
            "put_sample_on_incubator": self.put_sample_on_incubator,
            "transport_from_incubator_to_microscope1": self.transport_from_incubator_to_microscope1,
            "transport_from_microscope1_to_incubator": self.transport_from_microscope1_to_incubator,
            "connect": self.connect,
            "disconnect": self.disconnect,
            "halt": self.halt,
            "get_all_joints": self.get_all_joints,
            "get_all_positions": self.get_all_positions,
            # Add status functions
            "get_task_status": self.get_task_status,
            "reset_task_status": self.reset_task_status,
            "reset_all_task_status": self.reset_all_task_status,
            "light_on": self.light_on,
            "light_off": self.light_off
        })

        print(f"Robotic arm control service registered at workspace: {server.config.workspace}, id: {svc.id}")
        print(f'You can use this service using the service id: {svc.id}')
        id = svc.id.split(":")[1]
        print(f"You can also test the service via the HTTP proxy: {self.server_url}/{server.config.workspace}/services/{id}/move_sample_from_microscope1_to_incubator")

    async def setup(self):
        if self.local:
            token = None
            server = await connect_to_server({"server_url": self.server_url, "token": token, "ping_interval": None})
        else:
            try:
                token = os.environ.get("REEF_WORKSPACE_TOKEN")
            except:
                token = await login({"server_url": self.server_url})
            server = await connect_to_server({"server_url": self.server_url, "token": token, "workspace": "reef-imaging", "ping_interval": None})

        await self.start_hypha_service(server)

    def get_task_status(self, task_name):
        """Get the status of a specific task"""
        return self.task_status.get(task_name, "unknown")

    def reset_task_status(self, task_name):
        """Reset the status of a specific task"""
        if task_name in self.task_status:
            self.task_status[task_name] = "not_started"
    
    def reset_all_task_status(self):
        """Reset the status of all tasks"""
        for task_name in self.task_status:
            self.task_status[task_name] = "not_started"

    @schema_function(skip_self=True)
    def connect(self):
        """
        Connect and occupy the robot, so that it can be controlled.
        Returns: bool
        """
        self.task_status["connect"] = "started"
        try:
            self.robot.connect(self.ip)
            self.connected = True
            self.task_status["connect"] = "finished"
            print("Connected to robot")
            return True
        except Exception as e:
            self.task_status["connect"] = "failed"
            print(f"Failed to connect: {e}")
            return False

    @schema_function(skip_self=True)
    def disconnect(self):
        """
        Disconnect the robot, so that it can be used by other clients.
        Returns: bool
        """
        self.task_status["disconnect"] = "started"
        try:
            self.robot.close()
            self.connected = False
            self.task_status["disconnect"] = "finished"
            print("Disconnected from robot")
            return True
        except Exception as e:
            self.task_status["disconnect"] = "failed"
            print(f"Failed to disconnect: {e}")
            return False

    @schema_function(skip_self=True)
    def set_motor(self, state: int=Field(1, description="Enable or disable the motor, 1 for enable, 0 for disable")):
        if not self.connected:
            self.connect()
        self.robot.set_motor(state)
        return f"Motor set to {state}"

    @schema_function(skip_self=True)
    def play_script(self, script_path):
        if not self.connected:
            self.connect()
        result = self.robot.play_script(script_path)
        if result != 2:
            raise Exception("Error playing script")
        else:
            return "Script played"
    
    @schema_function(skip_self=True)
    def get_all_joints(self):
        """
        Get the current position of all joints
        Returns: dict
        """
        self.task_status["get_all_joints"] = "started"
        try:
            if not self.connected:
                self.connect()
            result = self.robot.get_all_joint()
            self.task_status["get_all_joints"] = "finished"
            return result
        except Exception as e:
            self.task_status["get_all_joints"] = "failed"
            print(f"Failed to get all joints: {e}")
            return {}

    @schema_function(skip_self=True)
    def get_all_positions(self):
        """
        Get the current position of all joints
        Returns: dict
        """
        self.task_status["get_all_positions"] = "started"
        try:
            if not self.connected:
                self.connect()
            result = self.robot.get_all_pose()
            self.task_status["get_all_positions"] = "finished"
            return result
        except Exception as e:
            self.task_status["get_all_positions"] = "failed"
            print(f"Failed to get all positions: {e}")
            return {}

    @schema_function(skip_self=True)
    def move_sample_from_microscope1_to_incubator(self):
        """
        Move sample from microscope1 to incubator, the microscope need to be homed before
        Returns: bool
        """
        task_name = "move_sample_from_microscope1_to_incubator"
        self.task_status[task_name] = "started"
        self.set_motor(1)
        try:
            self.play_script("paths/microscope1_to_incubator.txt")
            print("Sample moved from microscope1 to incubator")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to move sample from microscope1 to incubator: {e}")
            return False

    @schema_function(skip_self=True)
    def move_sample_from_incubator_to_microscope1(self):
        """
        Move sample from incubator to microscope1, microscope need to be homed before
        Returns: bool
        """
        task_name = "move_sample_from_incubator_to_microscope1"
        self.task_status[task_name] = "started"
        self.set_motor(1)
        try:
            self.play_script("paths/incubator_to_microscope1.txt")
            print("Sample moved from incubator to microscope1")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to move sample from incubator to microscope1: {e}")
            return False

    @schema_function(skip_self=True)
    def grab_sample_from_microscope1(self):
        """
        Transport a sample from microscope1 to the incubator
        Returns: bool
        """
        task_name = "grab_sample_from_microscope1"
        self.task_status[task_name] = "started"
        self.set_motor(1)
        try:
            self.play_script("paths/grab_from_microscope1.txt")
            print("Sample grabbed from microscope1")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to grab sample from microscope1: {e}")
            return False

    @schema_function(skip_self=True)
    def grab_sample_from_incubator(self):
        """
        Grab a sample from the incubator
        Returns: bool
        """
        task_name = "grab_sample_from_incubator"
        self.task_status[task_name] = "started"
        self.set_motor(1)
        try:
            self.play_script("paths/grab_from_incubator.txt")
            print("Sample grabbed from incubator")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to grab sample from incubator: {e}")
            return False

    @schema_function(skip_self=True)
    def put_sample_on_microscope1(self):
        """
        Place a sample on microscope1
        Returns: bool
        """
        task_name = "put_sample_on_microscope1"
        self.task_status[task_name] = "started"
        self.set_motor(1)
        try:
            self.play_script("paths/put_on_microscope1.txt")
            print("Sample placed on microscope1")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to put sample on microscope1: {e}")
            return False

    @schema_function(skip_self=True)
    def put_sample_on_incubator(self):
        """
        Place a sample on the incubator.
        Returns: bool
        """
        task_name = "put_sample_on_incubator"
        self.task_status[task_name] = "started"
        self.set_motor(1)
        try:
            self.play_script("paths/put_on_incubator.txt")
            print("Sample placed on incubator")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to put sample on incubator: {e}")
            return False

    @schema_function(skip_self=True)
    def transport_from_incubator_to_microscope1(self):
        """
        Transport a sample from the incubator to microscope1
        Returns: bool
        """
        task_name = "transport_from_incubator_to_microscope1"
        self.task_status[task_name] = "started"
        self.set_motor(1)
        try:
            self.play_script("paths/transport_from_incubator_to_microscope1.txt")
            print("Sample moved from incubator to microscope1")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to transport sample from incubator to microscope1: {e}")
            return False

    @schema_function(skip_self=True)
    def transport_from_microscope1_to_incubator(self):
        """
        Transport a sample from microscope1 to the incubator
        Returns: bool
        """
        task_name = "transport_from_microscope1_to_incubator"
        self.task_status[task_name] = "started"
        self.set_motor(1)
        try:
            self.play_script("paths/transport_from_microscope1_to_incubator.txt")
            print("Sample moved from microscope1 to incubator")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to transport sample from microscope1 to incubator: {e}")
            return False

    @schema_function(skip_self=True)
    def halt(self):
        """
        Halt/stop the robot, stop all the movements
        Returns: bool
        """
        task_name = "halt"
        self.task_status[task_name] = "started"
        try:
            if not self.connected:
                self.connect()
            self.robot.halt()
            print("Robot halted")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to halt robot: {e}")
            return False
    
    @schema_function(skip_self=True)
    def light_on(self):
        """
        Turn on the light
        """
        task_name = "light_on"
        self.task_status[task_name] = "started"
        try:
            self.robot.set_output(7, 0)
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to turn on light: {e}")
            return False

    @schema_function(skip_self=True)
    def light_off(self):    
        """
        Turn off the light
        """
        task_name = "light_off"
        self.task_status[task_name] = "started"
        try:
            self.robot.set_output(7, 1)
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to turn off light: {e}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Hypha service for the robotic arm.")
    parser.add_argument('--local', action='store_true', help="Use localhost as server URL")
    args = parser.parse_args()

    robotic_arm_service = RoboticArmService(local=args.local)

    loop = asyncio.get_event_loop()

    async def main():
        try:
            await robotic_arm_service.setup()
        except Exception as e:
            print(f"Error setting up robotic arm service: {e}")
            raise e

    loop.create_task(main())
    loop.run_forever()
