import asyncio
import time
import base64
from IPython.display import Image, display
from hypha_rpc import connect_to_server, login
import os
import dotenv
import logging  # Added for better error logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

dotenv.load_dotenv()
ENV_FILE = dotenv.find_dotenv()
if ENV_FILE:
    dotenv.load_dotenv(ENV_FILE)

server_url = "https://hypha.aicell.io"

reef_token = os.environ.get("REEF_WORKSPACE_TOKEN")
squid_token = os.environ.get("SQUID_WORKSPACE_TOKEN")

# Global variables for the services and state
incubator = None
microscope = None
robotic_arm = None
sample_loaded = False

async def setup_connections():
    global reef_token, squid_token, incubator, microscope, robotic_arm
    if not reef_token or not squid_token:
        token = await login({"server_url": server_url})
        reef_token = token
        squid_token = token

    reef_server = await connect_to_server({
        "server_url": server_url,
        "token": reef_token,
        "workspace": "reef-imaging",
        "ping_interval": None
    })
    squid_server = await connect_to_server({
        "server_url": server_url,
        "token": squid_token,
        "workspace": "squid-control",
        "ping_interval": None
    })

    incubator_id = "incubator-control"
    microscope_id = "microscope-control-squid-real-microscope-reef"
    robotic_arm_id = "robotic-arm-control"

    incubator = await reef_server.get_service(incubator_id)
    microscope = await squid_server.get_service(microscope_id)
    robotic_arm = await reef_server.get_service(robotic_arm_id)
    print('Connected to devices.')
    return incubator, microscope, robotic_arm

async def check_sample_loaded():
    global sample_loaded
    if sample_loaded:
        print("Sample plate has already been loaded onto the microscope")
    else:
        print("Sample plate is not loaded yet")
    return sample_loaded

async def load_plate_from_incubator_to_microscope(incubator_slot=10):
    global sample_loaded, incubator, microscope, robotic_arm
    try:
        assert not sample_loaded, "Sample plate has already been loaded"
        await incubator.get_sample_from_slot_to_transfer_station(incubator_slot)
        await microscope.home_stage()
        print("Plate loaded on station.")
        
        try:
            await robotic_arm.grab_sample_from_incubator()
            print("Sample grabbed.")
        except Exception as e:
            logger.error(f"Error grabbing sample from incubator: {e}")
            await robotic_arm.halt()
            # Return early to prevent further steps from executing
            return False
            
        try:
            await robotic_arm.transport_from_incubator_to_microscope1()
            print("Sample transported.")
        except Exception as e:
            robotic_arm.halt()
            logger.error(f"Error transporting sample to microscope: {e}")
            # Attempt to return sample to incubator if possible
            try:
                await robotic_arm.transport_from_microscope1_to_incubator()
                await robotic_arm.put_sample_on_incubator()
                await incubator.put_sample_from_transfer_station_to_slot(incubator_slot)
            except:
                robotic_arm.halt()
                logger.error("Failed to return sample to incubator after transport failure")
            return False
            
        # Continue with remaining operations with error handling
        try:
            await robotic_arm.put_sample_on_microscope1()
            print("Sample placed on microscope.")
        except Exception as e:
            robotic_arm.halt()
            logger.error(f"Error placing sample on microscope: {e}")
            return False
            
        await microscope.return_stage()
        print("Sample plate successfully loaded onto microscope stage.")
        sample_loaded = True
        return True
    except Exception as e:
        logger.error(f"Error during sample loading process: {e}")
        return False

async def unload_plate_from_microscope(incubator_slot=10):
    global sample_loaded, incubator, microscope, robotic_arm
    try:
        assert sample_loaded, "Sample plate is not on the microscope"
        await microscope.home_stage()
        print("Microscope homed.")
        
        try:
            await robotic_arm.grab_sample_from_microscope1()
            print("Sample grabbed from microscope.")
        except Exception as e:
            robotic_arm.halt()
            logger.error(f"Error grabbing sample from microscope: {e}")
            return False
            
        try:
            await robotic_arm.transport_from_microscope1_to_incubator()
            print("Sample moved to incubator.")
        except Exception as e:
            logger.error(f"Error transporting sample to incubator: {e}")
            return False
            
        try:
            await robotic_arm.put_sample_on_incubator()
            print("Sample placed on incubator.")
        except Exception as e:
            logger.error(f"Error placing sample on incubator: {e}")
            return False
            
        await incubator.put_sample_from_transfer_station_to_slot(incubator_slot)
        print("Sample moved to incubator.")
        await microscope.return_stage()
        print("Sample successfully unloaded from the microscopy stage.")
        sample_loaded = False
        return True
    except Exception as e:
        logger.error(f"Error during sample unloading process: {e}")
        return False

async def run_cycle():
    """Run the complete load-scan-unload process."""
    try:
        loading_success = await load_plate_from_incubator_to_microscope(incubator_slot=10)
        if not loading_success:
            logger.error("Failed to load sample - aborting cycle")
            return False
            
        try:
            await microscope.scan_well_plate(
                illuminate_channels=['BF LED matrix full','Fluorescence 488 nm Ex','Fluorescence 561 nm Ex'],
                do_reflection_af=True,
                scanning_zone=[(0,0),(7,11)], 
                action_ID='20250313'
            )
        except Exception as e:
            logger.error(f"Error during microscope scanning: {e}")
            # Even if scanning fails, try to return the sample to incubator
        
        unloading_success = await unload_plate_from_microscope(incubator_slot=10)
        if not unloading_success:
            logger.error("Failed to unload sample - manual intervention may be required")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Unexpected error during cycle: {e}")
        return False

async def run_time_lapse(round_time=3600):
    """Run the cycle every hour (xxx seconds)."""

    while True:

        # set up connection every time
        global incubator, microscope, robotic_arm
        incubator, microscope, robotic_arm = await setup_connections()
        start_time = asyncio.get_event_loop().time()
        logger.info("Starting new cycle...")
        success = await run_cycle()
        if success:
            logger.info("Cycle completed successfully")
        else:
            logger.warning("Cycle completed with errors")
            
        end_time = asyncio.get_event_loop().time()
        elapsed = end_time - start_time
        sleep_time = max(0, round_time - elapsed)
        print(f"Elapsed time: {elapsed:.2f} seconds. Waiting {sleep_time:.2f} seconds until next cycle.")
        await asyncio.sleep(sleep_time)

async def main():
    global incubator, microscope, robotic_arm
    incubator, microscope, robotic_arm = await setup_connections()
    await run_time_lapse(round_time=3600)

if __name__ == '__main__':
    asyncio.run(main())
