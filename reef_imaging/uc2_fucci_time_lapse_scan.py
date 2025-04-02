import asyncio
import time
import base64
from IPython.display import Image, display
from hypha_rpc import connect_to_server, login
import os
import dotenv
import logging
import sys
import logging.handlers

# Set up logging
def setup_logging(log_file="uc2_fucci_time_lapse_scan.log", max_bytes=100000, backup_count=3):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

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

# Configuration settings
IMAGING_INTERVAL = 3600  # Time between cycles in seconds
INCUBATOR_SLOT = 36  # Slot number in the incubator
ILLUMINATE_CHANNELS = ['BF LED matrix full', 'Fluorescence 488 nm Ex', 'Fluorescence 561 nm Ex']
SCANNING_ZONE = [(0, 0), (7, 11)]
Nx = 3
Ny = 3
ACTION_ID = '20250327-after-drug-treatment'

async def setup_connections(max_retries=300, timeout=10):
    """Set up connections to all services with retry mechanism."""
    global reef_token, squid_token, incubator, microscope, robotic_arm
    retries = 0
    
    while retries < max_retries:
        try:
            if not reef_token or not squid_token:
                token = await login({"server_url": server_url})
                reef_token = token
                squid_token = token

            reef_server = await asyncio.wait_for(
                connect_to_server({
                    "server_url": server_url,
                    "token": reef_token,
                    "workspace": "reef-imaging",
                    "ping_interval": None
                }),
                timeout=timeout
            )
            
            squid_server = await asyncio.wait_for(
                connect_to_server({
                    "server_url": server_url,
                    "token": squid_token,
                    "workspace": "squid-control",
                    "ping_interval": None
                }),
                timeout=timeout
            )

            incubator_id = "incubator-control"
            microscope_id = "microscope-control-squid-real-microscope-reef"
            robotic_arm_id = "robotic-arm-control"

            incubator = await asyncio.wait_for(reef_server.get_service(incubator_id), timeout=timeout)
            microscope = await asyncio.wait_for(squid_server.get_service(microscope_id), timeout=timeout)
            robotic_arm = await asyncio.wait_for(reef_server.get_service(robotic_arm_id), timeout=timeout)
            
            logger.info('Successfully connected to all devices.')
            return incubator, microscope, robotic_arm
            
        except asyncio.TimeoutError:
            logger.warning(f"Connection attempt {retries + 1}/{max_retries} timed out. Retrying...")
        except Exception as e:
            logger.error(f"Error during connection setup: {e}. Retrying... ({retries + 1}/{max_retries})")
        
        retries += 1
        if retries < max_retries:
            await asyncio.sleep(timeout)
    
    logger.error("Failed to establish connections after maximum retries. Terminating.")
    raise ConnectionError("Failed to establish connections to required services")

async def disconnect_services():
    """Disconnect from all services."""
    global incubator, microscope, robotic_arm
    try:
        if incubator:
            await incubator.disconnect()
        if microscope:
            await microscope.disconnect()
        if robotic_arm:
            await robotic_arm.disconnect()
        print('Disconnected from devices.')
    except Exception as e:
        logger.error(f"Error during disconnection: {e}")

async def call_service_with_retries(service, method_name, *args, max_retries=30, timeout=30):
    retries = 0
    while retries < max_retries:
        try:
            # Check the status of the task
            status = await service.get_task_status(method_name)
            logger.info(f"Task {method_name} status: {status}")
            if status == "failed":
                message = f"Task {method_name} failed. Stopping execution."
                logger.error(message)
                return False

            if status == "not_started":
                logger.info(f"Starting the task {method_name}...")
                await asyncio.wait_for(getattr(service, method_name)(*args), timeout=timeout)

            # Wait for the task to complete
            while True:
                status = await service.get_task_status(method_name)
                logger.info(f"Task {method_name} status: {status}")
                if status == "finished":
                    logger.info(f"Task {method_name} completed successfully.")
                    await service.reset_task_status(method_name)
                    return True
                elif status == "failed":
                    logger.error(f"Task {method_name} failed.")
                    return False
                await asyncio.sleep(1)

        except asyncio.TimeoutError:
            logger.warning(f"Operation {method_name} timed out. Retrying... ({retries + 1}/{max_retries})")
        except Exception as e:
            logger.error(f"Error: {e}. Retrying... ({retries + 1}/{max_retries})")
        retries += 1
        await asyncio.sleep(timeout)

    logger.error(f"Max retries reached for task {method_name}. Terminating.")
    return False

async def load_plate_from_incubator_to_microscope(incubator_slot=INCUBATOR_SLOT):
    global sample_loaded, incubator, microscope, robotic_arm
    if sample_loaded:
        logger.info("Sample plate has already been loaded onto the microscope")
        return True

    logger.info(f"Loading sample from incubator slot {incubator_slot} to transfer station...")
    if not await call_service_with_retries(incubator, "get_sample_from_slot_to_transfer_station", incubator_slot, timeout=60):
        return False

    logger.info(f"Homing the microscope stage...")
    if not await call_service_with_retries(microscope, "home_stage", timeout=30):
        return False

    logger.info(f"Grabbing sample from incubator...")
    if not await call_service_with_retries(robotic_arm, "grab_sample_from_incubator", timeout=120):
        return False

    logger.info(f"Transporting sample from incubator to microscope...")
    if not await call_service_with_retries(robotic_arm, "transport_from_incubator_to_microscope1", timeout=120):
        return False

    logger.info(f"Putting sample on microscope...")
    if not await call_service_with_retries(robotic_arm, "put_sample_on_microscope1", timeout=120):
        return False

    logger.info(f"Returning microscope stage to loading position...")
    if not await call_service_with_retries(microscope, "return_stage", timeout=30):
        return False

    logger.info("Sample plate successfully loaded onto microscope stage.")
    sample_loaded = True
    return True

async def unload_plate_from_microscope(incubator_slot=INCUBATOR_SLOT):
    global sample_loaded, incubator, microscope, robotic_arm
    if not sample_loaded:
        logger.info("Sample plate is not on the microscope")
        return True

    logger.info(f"Homing the microscope stage...")
    if not await call_service_with_retries(microscope, "home_stage", timeout=30):
        return False

    logger.info(f"Grabbing sample from microscope...")
    if not await call_service_with_retries(robotic_arm, "grab_sample_from_microscope1", timeout=120):
        return False

    logger.info(f"Transporting sample from microscope to incubator...")
    if not await call_service_with_retries(robotic_arm, "transport_from_microscope1_to_incubator", timeout=120):
        return False

    logger.info(f"Putting sample on incubator...")
    if not await call_service_with_retries(robotic_arm, "put_sample_on_incubator", timeout=120):
        return False

    logger.info(f"Putting sample on incubator slot {incubator_slot}...")    
    if not await call_service_with_retries(incubator, "put_sample_from_transfer_station_to_slot", incubator_slot, timeout=60):
        return False

    logger.info(f"Returning microscope stage to loading position...")
    if not await call_service_with_retries(microscope, "return_stage", timeout=30):
        return False

    logger.info("Sample successfully unloaded from the microscopy stage.")
    sample_loaded = False
    return True

async def run_cycle():
    """Run the complete load-scan-unload process."""

    #reset all task status
    await robotic_arm.reset_all_task_status()
    await incubator.reset_all_task_status()
    await microscope.reset_all_task_status()

    if not await load_plate_from_incubator_to_microscope(incubator_slot=INCUBATOR_SLOT):
        logger.error("Failed to load sample - aborting cycle")
        return False

    try:
        await asyncio.wait_for(
            microscope.scan_well_plate(
                illuminate_channels=ILLUMINATE_CHANNELS,
                do_reflection_af=True,
                scanning_zone=SCANNING_ZONE,
                Nx=Nx,
                Ny=Ny,
                action_ID=ACTION_ID
            ),
            timeout=2400
        )
    except asyncio.TimeoutError:
        logger.error("Microscope scanning timed out.")
    except Exception as e:
        logger.error(f"Error during microscope scanning: {e}")
        return False

    if not await unload_plate_from_microscope(incubator_slot=INCUBATOR_SLOT):
        logger.error("Failed to unload sample - manual intervention may be required")
        return False

    return True

async def run_time_lapse(round_time=IMAGING_INTERVAL):
    """Run the cycle every hour (xxx seconds)."""
    while True:
        global incubator, microscope, robotic_arm
        incubator, microscope, robotic_arm = await setup_connections()
        start_time = asyncio.get_event_loop().time()
        logger.info("Starting new cycle...")
        success = await run_cycle()
        if success:
            logger.info("Cycle completed successfully")
        else:
            logger.warning("Cycle completed with errors")
            await disconnect_services()
            sys.exit(1)

        await disconnect_services()
        end_time = asyncio.get_event_loop().time()
        elapsed = end_time - start_time
        sleep_time = max(0, round_time - elapsed)
        logger.info(f"Elapsed time: {elapsed:.2f} seconds. Waiting {sleep_time:.2f} seconds until next cycle.")
        await asyncio.sleep(sleep_time)

async def main():
    await run_time_lapse(round_time=IMAGING_INTERVAL)

if __name__ == '__main__':
    asyncio.run(main())
