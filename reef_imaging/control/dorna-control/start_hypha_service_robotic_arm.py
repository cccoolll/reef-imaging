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

    @schema_function(skip_self=True)
    async def connect(self):
        """
        Connect and occupy the robot, so that it can be controlled.
        Returns: bool
        """
        await asyncio.to_thread(self.robot.connect, self.ip)
        self.connected = True
        print("Connected to robot")
        return True

    @schema_function(skip_self=True)
    async def disconnect(self):
        """
        Disconnect the robot, so that it can be used by other clients.
        Returns: bool
        """
        await asyncio.to_thread(self.robot.close)
        self.connected = False
        print("Disconnected from robot")
        return True

    @schema_function(skip_self=True)
    async def set_motor(self, state: int=Field(1, description="Enable or disable the motor, 1 for enable, 0 for disable")):
        if not self.connected:
            await self.connect()
        self.robot.set_motor(state)
        return f"Motor set to {state}"

    @schema_function(skip_self=True)
    async def play_script(self, script_path):
        if not self.connected:
            await self.connect()
        result = await asyncio.to_thread(self.robot.play_script, script_path)
        if result != 2:
            raise Exception("Error playing script")
        else:
            return "Script played"

    @schema_function(skip_self=True)
    def move_sample_from_microscope1_to_incubator(self):
        """
        Move sample from microscope1 to incubator, the microscope need to be homed before
        Returns: bool
        """
        self.set_motor(1)
        try:
            self.play_script("paths/microscope1_to_incubator.txt")
            print("Sample moved from microscope1 to incubator")
            return True
        except Exception as e:
            print(f"Failed to move sample from microscope1 to incubator: {e}")
            return False

    @schema_function(skip_self=True)
    def move_sample_from_incubator_to_microscope1(self):
        """
        Move sample from incubator to microscope1, microscope need to be homed before
        Returns: bool
        """
        self.set_motor(1)
        try:
            self.play_script("paths/incubator_to_microscope1.txt")
            print("Sample moved from incubator to microscope1")
            return True
        except Exception as e:
            print(f"Failed to move sample from incubator to microscope1: {e}")
            return False

    @schema_function(skip_self=True)
    def grab_sample_from_microscope1(self):
        """
        Transport a sample from microscope1 to the incubator
        Returns: bool
        """
        self.set_motor(1)
        try:
            self.play_script("paths/grab_from_microscope1.txt")
            print("Sample grabbed from microscope1")
            return True
        except Exception as e:
            print(f"Failed to grab sample from microscope1: {e}")
            return False

    @schema_function(skip_self=True)
    def grab_sample_from_incubator(self):
        """
        Grab a sample from the incubator
        Returns: bool
        """
        self.set_motor(1)
        try:
            self.play_script("paths/grab_from_incubator.txt")
            print("Sample grabbed from incubator")
            return True
        except Exception as e:
            print(f"Failed to grab sample from incubator: {e}")
            return False

    @schema_function(skip_self=True)
    def put_sample_on_microscope1(self):
        """
        Place a sample on microscope1
        Returns: bool
        """
        self.set_motor(1)
        try:
            self.play_script("paths/put_on_microscope1.txt")
            print("Sample placed on microscope1")
            return True
        except Exception as e:
            print(f"Failed to put sample on microscope1: {e}")
            return False

    @schema_function(skip_self=True)
    def put_sample_on_incubator(self):
        """
        Place a sample on the incubator.
        Returns: bool
        """
        self.set_motor(1)
        try:
            self.play_script("paths/put_on_incubator.txt")
            print("Sample placed on incubator")
            return True
        except Exception as e:
            print(f"Failed to put sample on incubator: {e}")
            return False

    @schema_function(skip_self=True)
    def transport_from_incubator_to_microscope1(self):
        """
        Transport a sample from the incubator to microscope1
        Returns: bool
        """
        self.set_motor(1)
        try:
            self.play_script("paths/transport_from_incubator_to_microscope1.txt")
            print("Sample moved from incubator to microscope1")
            return True
        except Exception as e:
            print(f"Failed to transport sample from incubator to microscope1: {e}")
            return False

    @schema_function(skip_self=True)
    def transport_from_microscope1_to_incubator(self):
        """
        Transport a sample from microscope1 to the incubator
        Returns: bool
        """
        self.set_motor(1)
        try:
            self.play_script("paths/transport_from_microscope1_to_incubator.txt")
            print("Sample moved from microscope1 to incubator")
            return True
        except Exception as e:
            print(f"Failed to transport sample from microscope1 to incubator: {e}")
            return False

    @schema_function(skip_self=True)
    def halt(self):
        """
        Halt/stop the robot, stop all the movements
        Returns: bool
        """
        if not self.connected:
            self.connect()
        self.robot.halt()
        print("Robot halted")
        return True

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
