import time
from hypha_rpc.sync import connect_to_server, login
import os
import dotenv

# ...existing code for environment setup...
dotenv.load_dotenv()
ENV_FILE = dotenv.find_dotenv()
if ENV_FILE:
    dotenv.load_dotenv(ENV_FILE)

server_url = "https://hypha.aicell.io"
try:
    reef_token = os.environ.get("REEF_WORKSPACE_TOKEN")
except Exception as e:
    reef_token = login({"server_url": server_url})

reef_server = connect_to_server({
    "server_url": server_url,
    "token": reef_token,
    "workspace": "reef-imaging",
    "ping_interval": None
})

robotic_arm_id = "robotic-arm-control"
robotic_arm = reef_server.get_service(robotic_arm_id)

print("Starting dorna stress test: transporting sample continuously...")
iteration = 0

while True:
    iteration += 1
    print(f"\nIteration {iteration}: Transporting sample from incubator to microscope")
    try:
        robotic_arm.connect()
        # Transport from incubator to microscope
        robotic_arm.grab_sample_from_incubator()
        print("Sample grabbed from incubator.")
        robotic_arm.transport_from_incubator_to_microscope1()
        print("Sample transported to microscope.")
        robotic_arm.put_sample_on_microscope1()
        print("Sample placed on microscope.")
        
        # Transport from microscope back to incubator
        print(f"Iteration {iteration}: Transporting sample from microscope to incubator")
        robotic_arm.grab_sample_from_microscope1()
        print("Sample grabbed from microscope.")
        robotic_arm.transport_from_microscope1_to_incubator()
        print("Sample transported to incubator.")
        robotic_arm.put_sample_on_incubator()
        print("Sample placed on incubator.")
    except Exception as e:
        print(f"Error during iteration {iteration}: {e}")
    finally:
        robotic_arm.disconnect()
    
    # Pause before next iteration
    time.sleep(2)
