# Description: Orchestrates the time-lapse imaging workflow by coordinating the control of multiple devices.
import time
from hypha_rpc.sync import connect_to_server
import logging
# Connect to Hypha services for device control
server_url = "http://192.168.2.1:9527"
cytomat_id = "ws-user-pine-bumper-79360175/izm2rHYVhXeUqPDcateLHt:incubator-control"
dorna_id = ""  # Add the Dorna service ID here
microscope_id = ""  # Add the microscope service ID here

server = connect_to_server({"server_url": server_url})

cytomat = server.get_service(cytomat_id)
logging.info(f"Cytomat service connected: {cytomat}")
dorna = server.get_service(dorna_id)
logging.info(f"Dorna service connected: {dorna}")
microscope = server.get_service(microscope_id)
logging.info(f"Microscope service connected: {microscope}")


def wait_for_user_start():
    # User loads sample, adjusts settings, then clicks "start scanning"
    input("Sample loaded and settings adjusted. Press Enter to start scanning...")

def perform_scanning_round(microscope):
    # Trigger the microscope to perform one scanning round.
    print("Starting microscope scanning...")
    microscope.start_scanning()      # API call to start scanning
    # Optionally, wait/poll until scanning is done.
    while not microscope.scanning_complete():
        time.sleep(1)  # Polling delay
    print("Microscope scanning round complete.")

def move_sample_between_devices(robotic_arm, source, destination):
    # Control the robotic arm to move the sample from source to destination.
    print(f"Moving sample from {source} to {destination}...")
    robotic_arm.move_plate(source, destination)
    print("Movement complete.")

def run_time_lapse_workflow(num_rounds, incubation_wait):
    # Instantiate device controllers
    microscope = MicroscopeController()
    robotic_arm = RoboticArmController()
    incubator = IncubatorController()
    
    # Initial user-triggered start
    wait_for_user_start()
    
    for round_number in range(1, num_rounds + 1):
        print(f"\n--- Round {round_number} ---")
        
        # Step 1: Scan on microscope
        perform_scanning_round(microscope)
        
        # Step 2: Move sample from microscope to incubator
        move_sample_between_devices(robotic_arm, source="microscope", destination="incubator")
        
        # Step 3: Load sample into incubator
        print("Loading sample into incubator...")
        incubator.load_sample()  # API call to load the sample
        print("Sample loaded into incubator.")
        
        # Step 4: Wait until next round (incubation period)
        print(f"Incubating sample for {incubation_wait} seconds...")
        time.sleep(incubation_wait)
        
        # Step 5: Release sample from incubator
        print("Releasing sample from incubator...")
        incubator.release_sample()
        
        # Step 6: Move sample from incubator back to microscope
        move_sample_between_devices(robotic_arm, source="incubator", destination="microscope")
    
    print("\nTime-lapse workflow finished.")

if __name__ == "__main__":
    # Parameters could be loaded from a config file or passed as arguments
    NUM_ROUNDS = 5             # Total number of time-lapse rounds
    INCUBATION_WAIT = 60       # Wait time (in seconds) between rounds
    
    run_time_lapse_workflow(NUM_ROUNDS, INCUBATION_WAIT)
