import asyncio
import argparse
import os
from hypha_rpc import connect_to_server, login
from cytomat import Cytomat
from pydantic import Field
from hypha_rpc.utils.schema import schema_function
from typing import Optional
import dotenv
import json

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
        self.c = Cytomat("/dev/ttyUSB1", json_path="/home/tao/workspace/cytomat-controller/docs/config.json")
        self.samples_file = "/home/tao/workspace/reef-imaging/reef_imaging/control/cytomat-control/samples.json"
        # Add task status tracking
        self.task_status = {
            "initialize": "not_started",
            "put_sample_from_transfer_station_to_slot": "not_started",
            "get_sample_from_slot_to_transfer_station": "not_started",
            "get_status": "not_started",
            "is_busy": "not_started",
            "reset_error_status": "not_started",
            "get_sample_status": "not_started",
            "get_temperature": "not_started",
            "get_co2_level": "not_started",
            "get_slot_information": "not_started"
        }

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
            "is_busy": self.is_busy,
            "reset_error_status": self.reset_error_status,
            "get_sample_status": self.get_sample_status,
            "get_temperature": self.get_temperature,
            "get_co2_level": self.get_co2_level,
            "get_slot_information": self.get_slot_information,
            # Add status functions
            "get_task_status": self.get_task_status,
            "reset_task_status": self.reset_task_status,
            "reset_all_task_status": self.reset_all_task_status
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

    def get_task_status(self, task_name):
        """Get the status of a specific task"""
        try:
            status = self.task_status.get(task_name, "unknown")
            print(f"Task {task_name} status: {status}")  # Add logging
            return status
        except Exception as e:
            print(f"Error getting task status for {task_name}: {e}")
            return "unknown"

    def reset_task_status(self, task_name):
        """Reset the status of a specific task"""
        if task_name in self.task_status:
            self.task_status[task_name] = "not_started"

    def reset_all_task_status(self):
        """Reset the status of all tasks"""
        for task_name in self.task_status:
            self.task_status[task_name] = "not_started"

    @schema_function(skip_self=True)
    def initialize(self):
        """
        Clean up error status and initialize the incubator
        Returns:A string message        
        """
        task_name = "initialize"
        self.task_status[task_name] = "started"
        try:
            self.c.plate_handler.initialize()
            self.c.wait_until_not_busy(timeout=60)
            assert self.c.error_status == 0, f"Error status: {ERROR_CODES[self.c.error_status]}"
            self.task_status[task_name] = "finished"
            return "Incubator initialized."
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to initialize incubator: {e}")
            return f"Failed to initialize incubator: {e}"
    
    @schema_function(skip_self=True)
    def get_temperature(self):
        """Get the current temperature of the incubator"""
        task_name = "get_temperature"
        self.task_status[task_name] = "started"
        try:
            self.c.wait_until_not_busy(timeout=50)
            temperature = self.c.climate_controller.current_temperature
            print(f"Temperature: {temperature}")
            self.task_status[task_name] = "finished"
            return temperature
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to get temperature: {e}")
            return None
    
    @schema_function(skip_self=True)
    def get_slot_information(self, slot: int = Field(1, description="Slot number, range: 1-42")):
        """Get the current slot information of the incubator"""
        task_name = "get_slot_information"
        self.task_status[task_name] = "started"
        try:
            with open(self.samples_file, 'r') as file:
                samples = json.load(file)
            slot_info = next((sample for sample in samples if sample["incubator_slot"] == slot), None)
            self.task_status[task_name] = "finished"
            return slot_info
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to get slot information: {e}")
            return None

    @schema_function(skip_self=True)
    def get_co2_level(self):
        """Get the current CO2 level of the incubator"""
        task_name = "get_co2_level"
        self.task_status[task_name] = "started"
        try:
            self.c.wait_until_not_busy(timeout=50)
            co2_level = self.c.climate_controller.current_co2
            print(f"CO2 level: {co2_level}")
            self.task_status[task_name] = "finished"
            return co2_level
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to get CO2 level: {e}")
            return None

    @schema_function(skip_self=True)
    def reset_error_status(self):
        """Reset the error status of the incubator"""
        task_name = "reset_error_status"
        self.task_status[task_name] = "started"
        try:
            self.c.maintenance_controller.reset_error_status()
            self.task_status[task_name] = "finished"
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to reset error status: {e}")

    @schema_function(skip_self=True)
    def get_status(self):
        """
        Get the status of the incubator
        Returns: A dictionary
        """
        task_name = "get_status"
        self.task_status[task_name] = "started"
        try:
            status = {
                "error_status": self.c.error_status,
                "action_status": self.c.action_status,
                "busy": self.c.overview_status.busy
            }
            self.task_status[task_name] = "finished"
            return status
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to get status: {e}")
            return {}

    @schema_function(skip_self=True)
    def move_plate(self, slot:int=Field(5, description="Slot number,range: 1-42")):
        """
        Move plate from slot to transfer station and back
        Returns: A string message
        """
        task_name = "move_plate"
        self.task_status[task_name] = "started"
        try:
            c = self.c
            c.wait_until_not_busy(timeout=50)
            assert c.error_status == 0, f"Error status: {ERROR_CODES[self.c.error_status]}"
            c.plate_handler.move_plate_from_slot_to_transfer_station(slot)
            c.wait_until_not_busy(timeout=100)
            assert c.error_status == 0, f"Error status: {ERROR_CODES[self.c.error_status]}"
            c.plate_handler.move_plate_from_transfer_station_to_slot(slot)
            c.wait_until_not_busy(timeout=50)
            assert c.error_status == 0, f"Error status: {ERROR_CODES[self.c.error_status]}"
            self.task_status[task_name] = "finished"
            return f"Plate moved to slot {slot} and back to transfer station."
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to move plate: {e}")
            return f"Failed to move plate: {e}"
    
    def update_sample_status(self, slot: int, status: str):
        # Update the sample's status in samples.json
        with open(self.samples_file, 'r') as file:
            samples = json.load(file)
        for sample in samples:
            if sample["incubator_slot"] == slot:
                sample["status"] = status
                break
        with open(self.samples_file, 'w') as file:
            json.dump(samples, file, indent=4)

    @schema_function(skip_self=True)
    def put_sample_from_transfer_station_to_slot(self, slot: int = Field(5, description="Slot number,range: 1-42")):
        """
        Collect sample from transfer station to a slot
        Returns: A string message
        """
        task_name = "put_sample_from_transfer_station_to_slot"
        self.task_status[task_name] = "started"
        try:
            # Access status directly from JSON file
            with open(self.samples_file, 'r') as file:
                samples = json.load(file)
            current_status = next((sample["status"] for sample in samples if sample["incubator_slot"] == slot), None)
            assert current_status != "IN", "Plate is already inside the incubator"
            c = self.c
            c.plate_handler.move_plate_from_transfer_station_to_slot(slot)
            c.wait_until_not_busy(timeout=50)
            assert c.error_status == 0, f"Error status: {ERROR_CODES[self.c.error_status]}"
            self.update_sample_status(slot, "IN")
            self.task_status[task_name] = "finished"
            return f"Sample placed in slot {slot}."
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to put sample in slot {slot}: {e}")
            return f"Failed to put sample in slot {slot}: {e}"
    
    @schema_function(skip_self=True)
    def get_sample_from_slot_to_transfer_station(self, slot: int = Field(5, description="Slot number,range: 1-42")):
        """Release sample from a incubator's slot to it's transfer station."""
        task_name = "get_sample_from_slot_to_transfer_station"
        self.task_status[task_name] = "started"
        try:
            # Access status directly from JSON file
            with open(self.samples_file, 'r') as file:
                samples = json.load(file)
            current_status = next((sample["status"] for sample in samples if sample["incubator_slot"] == slot), None)
            assert current_status != "OUT", "Plate is already outside the incubator"
            c = self.c
            c.plate_handler.move_plate_from_slot_to_transfer_station(slot)
            c.wait_until_not_busy(timeout=50)
            assert c.error_status == 0, f"Error status: {ERROR_CODES[self.c.error_status]}"
            self.update_sample_status(slot, "OUT")
            self.task_status[task_name] = "finished"
            return f"Sample removed from slot {slot}."
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to get sample from slot {slot}: {e}")
            return f"Failed to get sample from slot {slot}: {e}"

    @schema_function(skip_self=True)
    def get_sample_status(self, slot: Optional[int] = Field(None, description="Slot number, range: 1-42, or None for all slots")):
        """Return the status of sample plates from samples.json"""
        task_name = "get_sample_status"
        self.task_status[task_name] = "started"
        try:
            with open(self.samples_file, 'r') as file:
                samples = json.load(file)
            
            if slot is None:
                status = {sample["incubator_slot"]: sample["status"] for sample in samples}
            else:
                status = next((sample["status"] for sample in samples if sample["incubator_slot"] == slot), "Unknown")
            
            self.task_status[task_name] = "finished"
            return status
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to get sample status: {e}")
            return {}

    def is_busy(self):
        task_name = "is_busy"
        self.task_status[task_name] = "started"
        try:
            busy = self.c.overview_status.busy
            self.task_status[task_name] = "finished"
            return busy
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to check if incubator is busy: {e}")
            return False

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