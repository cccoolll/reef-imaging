import asyncio
import argparse
import os
from hypha_rpc import connect_to_server, login
from dorna2 import Dorna
import dotenv

dotenv.load_dotenv()  
ENV_FILE = dotenv.find_dotenv()  
if ENV_FILE:  
    dotenv.load_dotenv(ENV_FILE)  

class DornaController:
    def __init__(self, ip="192.168.2.20"):
        self.robot = Dorna()
        self.ip = ip

    async def connect(self):
        await asyncio.to_thread(self.robot.connect, self.ip)
        print("Connected to robot")

    async def disconnect(self):
        await asyncio.to_thread(self.robot.close)
        print("Disconnected from robot")

    async def set_motor(self, state):
        await asyncio.to_thread(self.robot.set_motor, state)

    async def play_script(self, script_path):
        print("Playing script")
        await asyncio.to_thread(self.robot.play_script, script_path)

    async def is_busy(self):
        status = await asyncio.to_thread(self.robot.track_cmd)
        print(f"Robot status: {status}")
        return status["union"].get("stat", -1) != 2

    async def move_sample_from_microscope1_to_incubator(self):
        await self.set_motor(1)
        await self.play_script("paths/microscope1_to_incubator.txt")
    
    async def grab_sample_from_microscope1(self):
        await self.set_motor(1)
        await self.play_script("paths/grab_from_microscope1.txt")
    
    async def grab_sample_from_incubator(self):
        await self.set_motor(1)
        await self.play_script("paths/grab_from_incubator.txt")
    
    async def put_sample_on_microscope1(self):
        await self.set_motor(1)
        await self.play_script("paths/put_on_microscope1.txt")
    
    async def put_sample_on_incubator(self):
        await self.set_motor(1)
        await self.play_script("paths/put_on_incubator.txt")
    
    async def transport_from_incubator_to_microscope1(self):
        await self.set_motor(1)
        await self.play_script("paths/transport_from_incubator_to_microscope1.txt")
    
    async def transport_from_microscope1_to_incubator(self):
        await self.set_motor(1)
        await self.play_script("paths/transport_from_microscope1_to_incubator.txt")

    async def move_sample_from_incubator_to_microscope1(self):
        await self.set_motor(1)
        await self.play_script("paths/incubator_to_microscope1.txt")

    async def move_plate(self, source, destination):
        if source == "microscope" and destination == "incubator":
            await self.move_sample_from_microscope1_to_incubator()
        elif source == "incubator" and destination == "microscope1":
            await self.move_sample_from_incubator_to_microscope1()
        else:
            print(f"Invalid source-destination combination: {source} to {destination}")
    
    async def halt(self):
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

    async def move_sample_from_microscope1_to_incubator():
        await robotic_arm.move_sample_from_microscope1_to_incubator()
        print("Sample moved from microscope1 to incubator")
        return True
    
    async def move_sample_from_incubator_to_microscope1():
        await robotic_arm.move_sample_from_incubator_to_microscope1()
        print("Sample moved from incubator to microscope1")
        return True
    
    async def move_plate(source, destination):
        await robotic_arm.move_plate(source, destination)
        print(f"Sample moved from {source} to {destination}")
        return True
    
    async def grab_sample_from_microscope1():
        await robotic_arm.grab_sample_from_microscope1()
        print("Sample grabbed from microscope1")
        return True
    
    async def grab_sample_from_incubator():
        await robotic_arm.grab_sample_from_incubator()
        print("Sample grabbed from incubator")
        return True
    
    async def put_sample_on_microscope1():
        await robotic_arm.put_sample_on_microscope1()
        print("Sample placed on microscope1")
        return True
    
    async def put_sample_on_incubator():
        await robotic_arm.put_sample_on_incubator()
        print("Sample placed on incubator")
        return True
    
    async def transport_from_incubator_to_microscope1():
        await robotic_arm.transport_from_incubator_to_microscope1()
        print("Sample moved from incubator to microscope1")
        return True
    
    async def transport_from_microscope1_to_incubator():
        await robotic_arm.transport_from_microscope1_to_incubator()
        print("Sample moved from microscope1 to incubator")
        return True
    
    async def is_busy():
        """
        This function doesn't work, since dorna can't return the status of the robot during a move.
        """
        print("Checking if robotic arm is busy")
        return await robotic_arm.is_busy()
    
    async def connect():
        await robotic_arm.connect()
        print("Connected to robotic arm")
        return True

    async def disconnect():
        await robotic_arm.disconnect()
        print("Disconnected from robotic arm")
        return True
        
    async def halt():
        print("Halting robotic arm")
        await robotic_arm.halt()
        return True

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
        "move_plate": move_plate,
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
