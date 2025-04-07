# Description: Orchestrates the time-lapse imaging workflow by coordinating the control of multiple devices.
import time
import base64
from IPython.display import Image, display
from hypha_rpc.sync import connect_to_server, login
import os
import dotenv
import json

dotenv.load_dotenv()
ENV_FILE = dotenv.find_dotenv()
if ENV_FILE:
    dotenv.load_dotenv(ENV_FILE)

server_url = "https://hypha.aicell.io"
try:
    reef_token = os.environ.get("REEF_WORKSPACE_TOKEN")
    squid_token = os.environ.get("SQUID_WORKSPACE_TOKEN")
except:
    token = login({"server_url": server_url})
    reef_token = token
    squid_token = token

reef_server = connect_to_server({"server_url": server_url, "token": reef_token, "workspace": "reef-imaging", "ping_interval": None})
print(reef_server)
squid_server = connect_to_server({"server_url": server_url, "token": squid_token, "workspace": "squid-control", "ping_interval": None})

incubator_id = "incubator-control"
microscope_id = "microscope-control-squid-real-microscope-reef"
robotic_arm_id = "robotic-arm-control"

incubator = reef_server.get_service(incubator_id)
microscope = squid_server.get_service(microscope_id)
robotic_arm = reef_server.get_service(robotic_arm_id)

class TimeLapseOrchestrator:
    def __init__(self, config_path='config.json'):
        # Load sample configurations from JSON file
        with open(config_path, 'r') as file:
            self.config = json.load(file)

        # number of samples
        self.num_samples = len(self.config['samples'])
        print(f"Available samples: {self.num_samples}")

        # number of microscopes
        self.num_microscopes = len(self.config['microscopes'])
        print(f"Available microscopes: {self.num_microscopes}")

        self.server_url = server_url
        self.incubator_id = incubator_id
        self.robotic_arm_id = robotic_arm_id
        self.microscope_id = microscope_id

        self.server = reef_server
        self.incubator = incubator
        print(f"Cytomat service connected: {self.incubator}")
        self.robotic_arm = robotic_arm
        print(f"Dorna service connected: {self.robotic_arm}")
        self.microscope = microscope
        print(f"Microscope service connected: {self.microscope}")
    
    def complete_process_transport_sample_from_incubator_to_microscope(self, incubator_slot=3, microscope=1):
        self.robotic_arm.connect()
        # Move sample from incubator to microscope
        self.incubator.get_sample_from_slot_to_transfer_station(incubator_slot)
        print("Sample moved to transfer station.")
        while self.incubator.is_busy():
            time.sleep(1)
        self.microscope.home_stage()
        print("microscope homed.")
        if microscope == 1:
            self.robotic_arm.grab_sample_from_incubator()
            print("Sample grabbed from incubator.")
            self.robotic_arm.transport_from_incubator_to_microscope1()
            print("Sample moved to microscope.")
            self.robotic_arm.put_sample_on_microscope1()
            print("Sample placed on microscope.")
        else:
            print("Invalid microscope number.")
            return
        print("Sample moved to microscope.")
        self.microscope.return_stage()
        print("microscope returned.")

        self.robotic_arm.disconnect()
    
    def complete_process_transport_sample_from_microscope_to_incubator(self, microscope=1,incubator_slot=3):

        self.robotic_arm.connect()
        # Move sample from microscope to incubator
        self.microscope.home_stage()
        print("microscope homed.")
        if microscope == 1:
            self.robotic_arm.grab_sample_from_microscope1()
            print("Sample grabbed from microscope.")
            self.robotic_arm.transport_from_microscope1_to_incubator()
            print("Sample moved to incubator.")
            self.robotic_arm.put_sample_on_incubator()
            print("Sample placed on incubator.")
        else:
            print("Invalid microscope number.")
            return
        
        print("Sample moved from microscope1 to incubator")

        self.incubator.put_sample_from_transfer_station_to_slot(incubator_slot)
        while self.incubator.is_busy():
            time.sleep(1)
        print("Sample moved to incubator.")
        self.microscope.return_stage()

        self.robotic_arm.disconnect()

    def perform_scanning_round(self,do_reflection_af=True,scanning_zone=[(0,0),(7,11)], action_ID='testPlateScan'):
        # Trigger the microscope to perform one scanning round.
        print("Starting microscope scanning...")
        self.microscope.scan_well_plate(do_reflection_af=do_reflection_af,scanning_zone=scanning_zone, action_ID=action_ID)  # API call to start scanning
        # Optionally, wait/poll until scanning is done.
        print("Microscope scanning round complete.")

    def run_time_lapse_workflow(self, num_rounds, delta_t):
        # Initial user-triggered start
        start_time = time.time()
        print(f"Starting time-lapse workflow at {start_time}...")
        for round_number in range(1, num_rounds + 1):
            print(f"\n--- Round {round_number} ---")
            # Step 1: put sample from incubator to microscope
            self.complete_process_transport_sample_from_incubator_to_microscope(incubator_slot=33)
            # Step 2: Scan on microscope
            self.perform_scanning_round()

            # Step 3: put sample from microscope to incubator
            self.complete_process_transport_sample_from_microscope_to_incubator(incubator_slot=33)
            print("Sample moved from microscope to incubator.")
            # Step 4: Wait for next round
            while time.time() < start_time + round_number * delta_t:
                time.sleep(1)

        print("\nTime-lapse workflow finished.")

if __name__ == "__main__":
    # Parameters could be loaded from a config file or passed as arguments
    NUM_ROUNDS = 200  # Total number of time-lapse rounds
    DELTA_T = 3600  # Wait time (in seconds) between rounds

    orchestrator = TimeLapseOrchestrator()
    orchestrator.run_time_lapse_workflow(NUM_ROUNDS, DELTA_T)