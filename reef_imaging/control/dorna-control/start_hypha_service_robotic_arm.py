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

class DornaController:
    def __init__(self, ip="192.168.2.20"):
        self.robot = Dorna()
        self.ip = ip
        self.connected = False

    async def connect(self):
        await asyncio.to_thread(self.robot.connect, self.ip)
        self.connected = True
        print("Connected to robot")
        return f"Connected to robot"

    async def disconnect(self):
        await asyncio.to_thread(self.robot.close)
        self.connected = False
        print("Disconnected from robot")
        return f"Disconnected from robot"
    
    async def set_motor(self, state: int=Field(1, description="Enable or disable the motor, 1 for enable, 0 for disable")):
        await asyncio.to_thread(self.robot.set_motor, state)
        return f"Motor set to {state}"

    async def play_script(self, script_path):
        await asyncio.to_thread(self.robot.play_script, script_path)
        return "Script played"
    
    async def move_sample_from_microscope1_to_incubator(self):
        if not self.connected:
            await self.connect()
        await self.set_motor(1)
        await self.play_script("paths/microscope1_to_incubator.txt")
        return "Sample moved from microscope1 to incubator"
    
    async def grab_sample_from_microscope1(self):
        if not self.connected:
            await self.connect()
        await self.set_motor(1)
        await self.play_script("paths/grab_from_microscope1.txt")
        return "Sample grabbed from microscope1"

    async def grab_sample_from_incubator(self):
        if not self.connected:
            await self.connect()
        await self.set_motor(1)
        await self.play_script("paths/grab_from_incubator.txt")
        return "Sample grabbed from incubator"
    
    async def put_sample_on_microscope1(self):
        if not self.connected:
            await self.connect()
        await self.set_motor(1)
        await self.play_script("paths/put_on_microscope1.txt")
        return "Sample placed on microscope1"

    async def put_sample_on_incubator(self):
        if not self.connected:
            await self.connect()
        await self.set_motor(1)
        await self.play_script("paths/put_on_incubator.txt")
        return "Sample placed on incubator"
    
    async def transport_from_incubator_to_microscope1(self):
        if not self.connected:
            await self.connect()
        await self.set_motor(1)
        await self.play_script("paths/transport_from_incubator_to_microscope1.txt")
        return "Sample moved from incubator to microscope1"
    
    async def transport_from_microscope1_to_incubator(self):
        if not self.connected:
            await self.connect()
        await self.set_motor(1)
        await self.play_script("paths/transport_from_microscope1_to_incubator.txt")
        return "Sample moved from microscope1 to incubator"

    async def move_sample_from_incubator_to_microscope1(self):
        if not self.connected:
            await self.connect()
        await self.set_motor(1)
        await self.play_script("paths/incubator_to_microscope1.txt")
        return "Sample moved from incubator to microscope1"

    
    async def halt(self):
        if not self.connected:
            await self.connect()
        await asyncio.to_thread(self.robot.halt)
        print("Robot halted")

robotic_arm = DornaController()

async def start_server(server_url, local):
    if local:
        token = None
        server = await connect_to_server({"server_url": server_url, "token": token, "ping_interval": None})
    else:
        try:
            token = os.environ.get("REEF_WORKSPACE_TOKEN")
        except:
            token = await login({"server_url": server_url})
        server = await connect_to_server({"server_url": server_url, "token": token, "workspace": "reef-imaging", "ping_interval": None})

    @schema_function()
    async def move_sample_from_microscope1_to_incubator():
        """
        Move sample from microscope1 to incubator, the microscope need to be homed before
        Returns: bool True
        """
        await robotic_arm.move_sample_from_microscope1_to_incubator()
        print("Sample moved from microscope1 to incubator")
        return True
    
    @schema_function()
    async def move_sample_from_incubator_to_microscope1():
        """
        Move sample from incubator to microscope1, microscope need to be homed before
        Returns: bool True
        """
        await robotic_arm.move_sample_from_incubator_to_microscope1()
        print("Sample moved from incubator to microscope1")
        return True
    
    @schema_function()
    async def grab_sample_from_microscope1():
        """
        Description:
        Transport a sample from microscope1 to the incubator
        Returns: bool True
        """
        await robotic_arm.grab_sample_from_microscope1()
        print("Sample grabbed from microscope1")
        return True
    
    @schema_function()
    async def grab_sample_from_incubator():
        """
        Grab a sample from the incubator
        Returns: bool True
        """
        await robotic_arm.grab_sample_from_incubator()
        print("Sample grabbed from incubator")
        return True
    
    @schema_function()
    async def put_sample_on_microscope1():
        """
        Place a sample on microscope1
        Returns:bool True
        """
        await robotic_arm.put_sample_on_microscope1()
        print("Sample placed on microscope1")
        return True
        
    @schema_function()
    async def put_sample_on_incubator():
        """
        Place a sample on the incubator.
        Returns: bool True
        """
        await robotic_arm.put_sample_on_incubator()
        print("Sample placed on incubator")
        return True

    @schema_function()
    async def transport_from_incubator_to_microscope1():
        """
        Transport a sample from the incubator to microscope1
        Returns: bool True
        """
        await robotic_arm.transport_from_incubator_to_microscope1()
        print("Sample moved from incubator to microscope1")
        return True
    
    @schema_function()
    async def transport_from_microscope1_to_incubator():
        """
        Transport a sample from microscope1 to the incubator
        Returns: bool True
        """
        await robotic_arm.transport_from_microscope1_to_incubator()
        print("Sample moved from microscope1 to incubator")
        return True

    @schema_function()
    async def connect():
        """
        Connect and occupy the robot, so that it can be controlled.
        Returns: bool True
        """
        await robotic_arm.connect()
        print("Connected to robotic arm")
        return True

    @schema_function()
    async def disconnect():
        """
        Disconnect the robot, so that it can be used by other clients.
        Returns: bool True
        """
        await robotic_arm.disconnect()
        print("Disconnected from robotic arm")
        return True
    
    @schema_function()
    async def halt():
        """
        Description:
        Halt/stop the robot, stop all the movements
        Returns: bool True
        """
        print("Halting robotic arm")
        await robotic_arm.halt()
        return True
    
    await robotic_arm.connect()

    svc = await server.register_service({
        "name": "Robotic Arm Control",
        "id": "robotic-arm-control",
        "config": {
            "visibility": "public",
        },
        "move_sample_from_microscope1_to_incubator": move_sample_from_microscope1_to_incubator,
        "move_sample_from_incubator_to_microscope1": move_sample_from_incubator_to_microscope1,
        "grab_sample_from_microscope1": grab_sample_from_microscope1,
        "grab_sample_from_incubator": grab_sample_from_incubator,
        "put_sample_on_microscope1": put_sample_on_microscope1,
        "put_sample_on_incubator": put_sample_on_incubator,
        "transport_from_incubator_to_microscope1": transport_from_incubator_to_microscope1,
        "transport_from_microscope1_to_incubator": transport_from_microscope1_to_incubator,
        "connect": connect,
        "disconnect": disconnect,
        "halt": halt,
    })

    print(f"Incubator control service registered at workspace: {server.config.workspace}, id: {svc.id}")

    print(f'You can use this service using the service id: {svc.id}')
    id = svc.id.split(":")[1]
    print(f'You can use this service using the service id: {id}')
    # Keep the server running
    await server.serve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Hypha service for the robotic arm.")
    parser.add_argument('--local', action='store_true', help="Use localhost as server URL")
    args = parser.parse_args()

    server_url = "http://localhost:9527" if args.local else "https://hypha.aicell.io"

    asyncio.run(start_server(server_url, args.local))
