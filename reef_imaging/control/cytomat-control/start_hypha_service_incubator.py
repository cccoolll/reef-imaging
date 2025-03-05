import asyncio
import argparse
import os
from hypha_rpc import connect_to_server, login
from cytomat import Cytomat
from pydantic import Field
from hypha_rpc.utils.schema import schema_function

import dotenv

dotenv.load_dotenv()  
ENV_FILE = dotenv.find_dotenv()  
if ENV_FILE:  
    dotenv.load_dotenv(ENV_FILE)  


ERROR_CODES = {
    0: "No error",
    1: "Motor communication disrupted",
    2: "Plate not mounted on shovel",
    3: "Plate not dropped from shovel",
    4: "Shovel not extended",
    5: "Procedure timeout",
    6: "Transfer door not opened",
    7: "Transfer door not closed",
    8: "Shovel not retracted",
    10: "Step motor temperature too high",
    11: "Other step motor error",
    12: "Transfer station not rotated",
    13: "Heating or CO2 communication disrupted",
    14: "Shaker communication disrupted",
    15: "Shaker configuration out of order",
    16: "Shaker not started",
    19: "Shaker clamp not open",
    20: "Shaker clamp not closed",
    255: "Critical"
}

class IncubatorService:
    def __init__(self, local):
        self.local = local
        self.server_url = "http://localhost:9527" if local else "https://hypha.aicell.io"
        self.c = Cytomat("/dev/ttyUSB0", json_path="/home/tao/workspace/cytomat-controller/docs/config.json")

    async def start_hypha_service(self, server):
        svc = await server.register_service({
            "name": "Incubator Control",
            "id": "incubator-control",
            "config": {
                "visibility": "public",
                "run_in_executor": True
            },
            "initialize": self.initialize,
            "put_sample_from_transfer_station_to_slot": self.put_sample_from_transfer_station_to_slot,
            "get_sample_from_slot_to_transfer_station": self.get_sample_from_slot_to_transfer_station,
            "get_status": self.get_status,
            "is_busy": self.is_busy
        })

        print(f"Incubator control service registered at workspace: {server.config.workspace}, id: {svc.id}")
        print(f'You can use this service using the service id: {svc.id}')
        id = svc.id.split(":")[1]
        print(f"You can also test the service via the HTTP proxy: {self.server_url}/{server.config.workspace}/services/{id}/initialize")

    async def setup(self):
        self.c.maintenance_controller.reset_error_status()
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
    def initialize(self):
        """
        Description:
        Initialize the incubator, it only needs to be called when the incubator needs to be recalibrated.
        Returns:
        str, shows the result of the operation.
        
        """
        self.c.plate_handler.initialize()
        self.c.wait_until_not_busy(timeout=60)
        assert self.c.error_status == 0, f"Error status: {ERROR_CODES[self.c.error_status]}"
        return "Incubator initialized."

    @schema_function(skip_self=True)
    def get_status(self):
        """
        Description:
        Get the status of the incubator.
        Returns:
        dict, shows the status of the incubator.
        """
        return {"error_status": self.c.error_status, "action_status": self.c.action_status, "busy": self.c.overview_status.busy}

    @schema_function(skip_self=True)
    def move_plate(self, slot:int=Field(5, description="Slot number,range: 1-42")):
        """
        Description: 
        Move plate from slot to transfer station and back, is used just for testing.
        Returns: 
        str, shows the result of the operation.
        """
        c = self.c
        c.wait_until_not_busy(timeout=50)
        assert c.error_status == 0, f"Error status: {ERROR_CODES[self.c.error_status]}"
        c.wait_until_not_busy(timeout=50)
        assert c.error_status == 0, f"Error status: {ERROR_CODES[self.c.error_status]}"
        c.plate_handler.move_plate_from_slot_to_transfer_station(slot)
        c.wait_until_not_busy(timeout=100)
        assert c.error_status == 0, f"Error status: {ERROR_CODES[self.c.error_status]}"
        c.plate_handler.move_plate_from_transfer_station_to_slot(slot)
        c.wait_until_not_busy(timeout=50)

        assert c.error_status == 0, f"Error status: {ERROR_CODES[self.c.error_status]}"
        return f"Plate moved to slot {slot} and back to transfer station."
    
    @schema_function(skip_self=True)
    def put_sample_from_transfer_station_to_slot(self, slot:int=Field(5, description="Slot number,range: 1-42")):
        """
        Description:
        Collect sample from incubator's transfer station to it's slot.
        Returns:
        str, shows the result of the operation.
        """
        c = self.c
        c.plate_handler.move_plate_from_transfer_station_to_slot(slot)
        assert c.error_status == 0, f"Error status: {ERROR_CODES[self.c.error_status]}"

    @schema_function(skip_self=True)
    def get_sample_from_slot_to_transfer_station(self, slot:int=Field(5, description="Slot number,range: 1-42")):
        """Release sample from a incubator's slot to it's transfer station."""
        c = self.c
        c.plate_handler.move_plate_from_slot_to_transfer_station(slot)
        c.wait_until_not_busy(timeout=50)
        assert c.error_status == 0, f"Error status: {ERROR_CODES[self.c.error_status]}"

    def is_busy(self):
        c = self.c
        return c.overview_status.busy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Hypha service for the incubator.")
    parser.add_argument('--local', action='store_true', help="Use localhost as server URL")
    args = parser.parse_args()

    incubator_service = IncubatorService(local=args.local)

    loop = asyncio.get_event_loop()

    async def main():
        try:
            await incubator_service.setup()
        except Exception as e:
            print(f"Error setting up incubator service: {e}")
            raise e

    loop.create_task(main())
    loop.run_forever()
