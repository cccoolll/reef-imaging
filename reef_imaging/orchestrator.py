# Description: Orchestrates the time-lapse imaging workflow by coordinating the control of multiple devices.
import time
from hypha_rpc.sync import connect_to_server
import json

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

        self.server_url = "http://192.168.2.1:9527"
        self.incubator_id = self.config['incubator']['settings']['id']
        self.robotic_arm_id = self.config['robotic_arm']['settings']['id']
        self.microscope_id = self.config['microscopes'][0]['settings']['id']

        self.server = connect_to_server({"server_url": self.server_url})
        self.incubator = self.server.get_service(self.incubator_id)
        print(f"Cytomat service connected: {self.incubator}")
        self.robotic_arm = self.server.get_service(self.robotic_arm_id)
        print(f"Dorna service connected: {self.robotic_arm}")
        self.microscope = self.server.get_service(self.microscope_id)
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

    def perform_scanning_round(self):
        # Trigger the microscope to perform one scanning round.
        print("Starting microscope scanning...")
        self.microscope.scan_well_plate()  # API call to start scanning
        # Optionally, wait/poll until scanning is done.
        microscope_status = self.microscope.get_status()
        while microscope_status["is_busy"]:
            microscope_status = self.microscope.get_status()
            time.sleep(3)  # Polling delay
        print("Microscope scanning round complete.")

    def run_time_lapse_workflow(self, num_rounds, delta_t):
        # Initial user-triggered start
        start_time = time.time()
        print(f"Starting time-lapse workflow at {start_time}...")
        for round_number in range(1, num_rounds + 1):
            print(f"\n--- Round {round_number} ---")

            # Step 1: put sample from incubator to microscope
            print("Moving sample from incubator to microscope...")
            self.incubator.get_sample_from_slot_to_transfer_station()
            self.robotic_arm.move_sample_from_incubator_to_microscope1()
            print("Sample moved to microscope.")

            # Step 2: Scan on microscope
            self.perform_scanning_round()

            # Step 3: put sample from microscope to incubator
            print("Moving sample from microscope to incubator...")
            self.robotic_arm.move_sample_from_microscope1_to_incubator()
            print("Sample moved to incubator.")
            self.incubator.put_sample_from_transfer_station_to_slot()
            print("Sample moved to incubator.")

            # Step 4: Wait for next round
            while time.time() < start_time + round_number * delta_t:
                time.sleep(1)

        print("\nTime-lapse workflow finished.")

if __name__ == "__main__":
    # Parameters could be loaded from a config file or passed as arguments
    NUM_ROUNDS = 5  # Total number of time-lapse rounds
    DELTA_T = 1800  # Wait time (in seconds) between rounds

    orchestrator = TimeLapseOrchestrator()
    #orchestrator.run_time_lapse_workflow(NUM_ROUNDS, DELTA_T)
    orchestrator.transport_sample_from_microscope_to_incubator()
