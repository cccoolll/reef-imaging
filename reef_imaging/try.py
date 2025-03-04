# Description: Orchestrates the time-lapse imaging workflow by coordinating the control of multiple devices.
import time
from hypha_rpc.sync import connect_to_server
import logging
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

        self.server_url = "http://reef.dyn.scilifelab.se:9527"
        self.incubator_id = self.config['incubator']['settings']['id']
        self.robotic_arm_id = self.config['robotic_arm']['settings']['id']
        self.microscope_id = self.config['microscopes'][0]['settings']['id']

        self.server = connect_to_server({"server_url": self.server_url})
        self.incubator = self.server.get_service(self.incubator_id)
        logging.info(f"Cytomat service connected: {self.incubator}")
        self.robotic_arm = self.server.get_service(self.robotic_arm_id)
        logging.info(f"Dorna service connected: {self.robotic_arm}")
        self.microscope = self.server.get_service(self.microscope_id)
        logging.info(f"Microscope service connected: {self.microscope}")

    def perform_scanning_round(self,do_reflection_af):
        # Trigger the microscope to perform one scanning round.
        print("Starting microscope scanning...")
        self.microscope.scan_well_plate(do_reflection_af=do_reflection_af)  # API call to start scanning
        # Optionally, wait/poll until scanning is done.
        microscope_status = self.microscope.get_status()
        print("Waiting for microscope to finish scanning...")
        print(microscope_status)
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


            #  put sample from microscope to incubator
            print("Moving sample from microscope to incubator...")
            self.robotic_arm.move_sample_from_microscope_to_incubator()
            print("Sample moved to incubator.")
            self.incubator.put_sample_from_transfer_station_to_slot()
            print("Sample moved to incubator.")

            # Step 4: Wait for next round
            while time.time() < start_time + round_number * delta_t:
                time.sleep(1)

        print("\nTime-lapse workflow finished.")

if __name__ == "__main__":
    # Parameters could be loaded from a config file or passed as arguments
    NUM_ROUNDS = 1  # Total number of time-lapse rounds
    DELTA_T = 18  # Wait time (in seconds) between rounds

    orchestrator = TimeLapseOrchestrator()
    orchestrator.run_time_lapse_workflow(NUM_ROUNDS, DELTA_T)
