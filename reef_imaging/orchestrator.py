# Description: Orchestrates the time-lapse imaging workflow by coordinating the control of multiple devices.
import time
from hypha_rpc.sync import connect_to_server
import logging
import json
# Connect to Hypha services for device control
server_url = "http://192.168.2.1:9527"

# Load sample configurations from JSON file
with open('config.json', 'r') as file:
    config = json.load(file)

# number of samples
num_samples = len(config['samples'])
print(f"Available samples: {num_samples}")

# number of microscopes
num_microscopes = len(config['microscopes'])
print(f"Available microscopes: {num_microscopes}")

incubator_id = config['incubator']['settings']['id']
robotic_arm_id = config['robotic_arm']['settings']['id']
microscope_id = config['microscopes'][0]['settings']['id']


server = connect_to_server({"server_url": server_url})

incubator = server.get_service(incubator_id)
logging.info(f"Cytomat service connected: {incubator}")
robotic_arm = server.get_service(robotic_arm_id)
logging.info(f"Dorna service connected: {robotic_arm}")
microscope = server.get_service(microscope_id)
logging.info(f"Microscope service connected: {microscope}")



def perform_scanning_round(microscope):
    # Trigger the microscope to perform one scanning round.
    print("Starting microscope scanning...")
    microscope.scan_well_plate()      # API call to start scanning
    # Optionally, wait/poll until scanning is done.
    microscope_status = microscope.get_status()
    while microscope_status["is_busy"]:
        microscope_status = microscope.get_status()
        time.sleep(3)  # Polling delay
    print("Microscope scanning round complete.")


def run_time_lapse_workflow(num_rounds, delta_t):
    
    # Initial user-triggered start
    start_time = time.time()
    print(f"Starting time-lapse workflow at {start_time}...")
    for round_number in range(1, num_rounds + 1):
        print(f"\n--- Round {round_number} ---")
        
        # Step 1: put sample from incubator to microscope
        print("Moving sample from incubator to microscope...")
        #incubator.get_sample_from_slot_to_transfer_station()
        #robotic_arm.move_sample_from_incubator_to_microscope()
        print("Sample moved to microscope.")

        # Step 2: Scan on microscope
        perform_scanning_round(microscope)

        # Step 3: put sample from microscope to incubator
        print("Moving sample from microscope to incubator...")
        #robotic_arm.move_sample_from_microscope_to_incubator()
        print("Sample moved to incubator.")
        #incubator.put_sample_from_transfer_station_to_slot()
        print("Sample moved to incubator.")

        # Step 4: Wait for next round
        while time.time() < start_time + round_number * delta_t:
            time.sleep(1)

        
    print("\nTime-lapse workflow finished.")

if __name__ == "__main__":
    # Parameters could be loaded from a config file or passed as arguments
    NUM_ROUNDS = 5             # Total number of time-lapse rounds
    DELTA_T = 60       # Wait time (in seconds) between rounds
    
    run_time_lapse_workflow(NUM_ROUNDS, DELTA_T)
