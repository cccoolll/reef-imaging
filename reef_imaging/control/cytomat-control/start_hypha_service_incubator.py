import asyncio
import argparse
import os
from hypha_rpc import connect_to_server, login
from cytomat import Cytomat

import dotenv

dotenv.load_dotenv()  
ENV_FILE = dotenv.find_dotenv()  
if ENV_FILE:  
    dotenv.load_dotenv(ENV_FILE)  
    
class IncubatorService:
    def __init__(self, local):
        self.local = local
        self.server_url = "http://localhost:9527" if local else "https://hypha.aicell.io"

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
            "is_busy": self.is_busy
        })

        print(f"Incubator control service registered at workspace: {server.config.workspace}, id: {svc.id}")
        print(f'You can use this service using the service id: {svc.id}')
        id = svc.id.split(":")[1]
        print(f"You can also test the service via the HTTP proxy: {self.server_url}/{server.config.workspace}/services/{id}/initialize")

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

    def initialize(self):
        c = Cytomat("/dev/ttyUSB0", json_path="/home/tao/workspace/cytomat-controller/docs/config.json")
        c.plate_handler.initialize()
        
    def move_plate(self, slot):
        c = Cytomat("/dev/ttyUSB0", json_path="/home/tao/workspace/cytomat-controller/docs/config.json")
        c.wait_until_not_busy(timeout=50)
        c.plate_handler.initialize()
        c.wait_until_not_busy(timeout=50)
        c.plate_handler.move_plate_from_transfer_station_to_slot(slot)
        c.wait_until_not_busy(timeout=50)
        c.plate_handler.move_plate_from_slot_to_transfer_station(slot)
        c.wait_until_not_busy(timeout=50)
        return f"Plate moved to slot {slot} and back to transfer station."
    
    def put_sample_from_transfer_station_to_slot(self, slot=5):
        c = Cytomat("/dev/ttyUSB0")
        c.plate_handler.move_plate_from_transfer_station_to_slot(slot)

    def get_sample_from_slot_to_transfer_station(self, slot=5):
        c = Cytomat("/dev/ttyUSB0")
        c.plate_handler.move_plate_from_slot_to_transfer_station(slot)
    
    def is_busy(self):
        c = Cytomat("/dev/ttyUSB0")
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

    loop.create_task(main())
    loop.run_forever()
