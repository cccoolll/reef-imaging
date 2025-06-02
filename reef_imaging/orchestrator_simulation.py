import asyncio
import time
import base64
from IPython.display import Image, display
from hypha_rpc import connect_to_server, login
from hypha_rpc.utils.schema import schema_function
import os
import dotenv
import logging
import sys
import logging.handlers
from datetime import datetime, timezone, timedelta
import argparse
import json
import random # For simulating occasional failures
import copy # Added import

# Set up logging
def setup_logging(log_file="orchestrator_simulation.log", max_bytes=10*1024*1024, backup_count=5):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Rotating file handler with 10MB limit
    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


dotenv.load_dotenv()
ENV_FILE = dotenv.find_dotenv()
if ENV_FILE:
    dotenv.load_dotenv(ENV_FILE)


# Simulated Service IDs - these remain as they define the simulated backend services
SIM_INCUBATOR_ID = "incubator-control-simulation"
SIM_DEFAULT_MICROSCOPE_ID = "microscope-squid-reef" # Default or fallback if task doesn't specify
SIM_ROBOTIC_ARM_ID = "robotic-arm-control-simulation"

CONFIG_FILE_PATH = "config_simulation.json" # Using a separate config for simulation
CONFIG_READ_INTERVAL = 10 # Seconds, faster for simulation
MAX_CYCLE_RETRIES = 2 # Fewer retries for faster sim testing
CYCLE_RETRY_DELAY = timedelta(seconds=10) # Shorter for simulation testing
ORCHESTRATOR_LOOP_SLEEP = 5 # Seconds, faster for simulation

# --- Mock Hypha Services ---
class MockHyphaService:
    def __init__(self, service_id, service_type, loop):
        self.id = service_id
        self.service_type = service_type
        self._loop = loop
        self._tasks_status = {}
        self._method_call_counts = {}
        logger.info(f"MockHyphaService {self.id} ({self.service_type}) initialized.")

    async def hello_world(self):
        logger.debug(f"Mock {self.id}: hello_world() called.")
        await asyncio.sleep(0.01) # Simulate network delay
        return "Hello world"

    async def get_all_task_status(self):
        logger.debug(f"Mock {self.id}: get_all_task_status() called. Current: {self._tasks_status}")
        await asyncio.sleep(0.01)
        return self._tasks_status.copy()

    async def get_task_status(self, task_name):
        status = self._tasks_status.get(task_name, "not_started")
        logger.debug(f"Mock {self.id}: get_task_status({task_name}) -> {status}")
        await asyncio.sleep(0.01)
        return status

    async def reset_task_status(self, task_name):
        if task_name in self._tasks_status:
            logger.info(f"Mock {self.id}: reset_task_status({task_name}) from {self._tasks_status[task_name]}")
            del self._tasks_status[task_name]
        else:
            logger.debug(f"Mock {self.id}: reset_task_status({task_name}) - no active status to reset.")
        await asyncio.sleep(0.01)
        return True
        
    async def reset_all_task_status(self):
        logger.info(f"Mock {self.id}: reset_all_task_status(). Current: {self._tasks_status}")
        self._tasks_status.clear()
        await asyncio.sleep(0.01)
        return True

    async def disconnect(self):
        logger.info(f"MockHyphaService {self.id} ({self.service_type}): disconnect() called.")
        await asyncio.sleep(0.01)
        return True

    async def _simulate_method_call(self, method_name, duration=0.5, fail_sometimes=False, fail_rate=0.1):
        self._method_call_counts[method_name] = self._method_call_counts.get(method_name, 0) + 1
        logger.info(f"Mock {self.id}: Starting method {method_name} (Call #{self._method_call_counts[method_name]})")
        self._tasks_status[method_name] = "running"
        await asyncio.sleep(duration / 2) # Halfway through
        
        # Simulate occasional failure
        if fail_sometimes and random.random() < fail_rate:
            logger.warning(f"Mock {self.id}: Simulating FAILURE for method {method_name}")
            self._tasks_status[method_name] = "failed"
            return False # Indicate failure to caller if needed by call_service_with_retries logic

        logger.info(f"Mock {self.id}: Finishing method {method_name}")
        await asyncio.sleep(duration / 2) # Remaining duration
        self._tasks_status[method_name] = "finished"
        return True

class MockIncubator(MockHyphaService):
    def __init__(self, service_id, loop):
        super().__init__(service_id, "incubator", loop)

    async def get_sample_from_slot_to_transfer_station(self, slot, **kwargs):
        return await self._simulate_method_call("get_sample_from_slot_to_transfer_station", duration=1)

    async def put_sample_from_transfer_station_to_slot(self, slot, **kwargs):
        return await self._simulate_method_call("put_sample_from_transfer_station_to_slot", duration=1)

class MockRoboticArm(MockHyphaService):
    def __init__(self, service_id, loop):
        super().__init__(service_id, "robotic_arm", loop)

    async def grab_sample_from_incubator(self, **kwargs):
        return await self._simulate_method_call("grab_sample_from_incubator", duration=0.5)

    async def transport_from_incubator_to_microscope1(self, **kwargs):
        return await self._simulate_method_call("transport_from_incubator_to_microscope1", duration=1)

    async def put_sample_on_microscope1(self, **kwargs):
        return await self._simulate_method_call("put_sample_on_microscope1", duration=0.5)

    async def grab_sample_from_microscope1(self, **kwargs):
        return await self._simulate_method_call("grab_sample_from_microscope1", duration=0.5)

    async def transport_from_microscope1_to_incubator(self, **kwargs):
        return await self._simulate_method_call("transport_from_microscope1_to_incubator", duration=1)

    async def put_sample_on_incubator(self, **kwargs):
        return await self._simulate_method_call("put_sample_on_incubator", duration=0.5)


class MockMicroscope(MockHyphaService):
    def __init__(self, service_id, loop):
        super().__init__(service_id, "microscope", loop)

    async def home_stage(self, **kwargs):
        return await self._simulate_method_call("home_stage", duration=0.3)

    async def return_stage(self, **kwargs):
        return await self._simulate_method_call("return_stage", duration=0.3)

    async def scan_well_plate_simulated(self, illuminate_channels, do_reflection_af, scanning_zone, Nx, Ny, action_ID, **kwargs):
        logger.info(f"Mock {self.id}: scan_well_plate_simulated called with action_ID: {action_ID}, channels: {illuminate_channels}, zone: {scanning_zone}, Nx: {Nx}, Ny: {Ny}, AF: {do_reflection_af}")
        # Simulate a longer scan, maybe with a chance of failure
        return await self._simulate_method_call("scan_well_plate_simulated", duration=2, fail_sometimes=True, fail_rate=0.05) # 5% chance of scan failure

# --- End Mock Hypha Services ---

class OrchestrationSystem:
    def __init__(self, local=False):
        self.local = local # local flag might not mean much without real connections
        self.server_url = "mock_server_url" # Not used for connections
        self.local=local
        self.token = None
        self.workspace=None
        if self.local:
            self.token = os.environ.get("REEF_LOCAL_TOKEN")
            self.orchestrator_hypha_server_url = "http://localhost:9527" # Default
            self.workspace=os.environ.get("REEF_LOCAL_WORKSPACE")
        else:
            self.token = os.environ.get("REEF_WORKSPACE_TOKEN")
            self.orchestrator_hypha_server_url = "https://hypha.aicell.io"
            self.workspace="reef-imaging"
        
        self.orchestrator_hypha_service_id = "orchestrator-manager-simulation"
        self.orchestrator_hypha_server_connection = None

        # Instantiate MOCK services
        loop = asyncio.get_event_loop()
        self.incubator = MockIncubator(SIM_INCUBATOR_ID, loop)
        # Microscope is instantiated on demand based on task, but we can have a placeholder or default
        self.microscope = None # Will be a MockMicroscope instance
        self.robotic_arm = MockRoboticArm(SIM_ROBOTIC_ARM_ID, loop)
        
        self.sample_on_microscope_flag = False

        self.sim_incubator_id = SIM_INCUBATOR_ID # Retain for clarity if needed
        self.sim_robotic_arm_id = SIM_ROBOTIC_ARM_ID # Retain for clarity

        self.current_microscope_id = None # Logical ID of the microscope for the current task
        self.current_mock_microscope_instance = None # Stores the active MockMicroscope
        
        self.tasks = {}
        self.health_check_tasks = {}
        self.active_task_name = None
        self._config_lock = asyncio.Lock() # Lock for reading/writing config file

    async def _register_self_as_hypha_service(self):
        logger.info(f"Registering orchestrator as a Hypha service with ID '{self.orchestrator_hypha_service_id}' on server '{self.orchestrator_hypha_server_url}'")
        try:
            token = None
            if self.local:
                token = os.environ.get("REEF_LOCAL_TOKEN")
            else:
                token = os.environ.get("ORCHESTRATOR_WORKSPACE_TOKEN", os.environ.get("REEF_WORKSPACE_TOKEN"))

            server_config = {"server_url": self.orchestrator_hypha_server_url, "ping_interval": None, "workspace": self.workspace}
            if token:
                server_config["token"] = token
            self.orchestrator_hypha_server_connection = await connect_to_server(server_config)
            logger.info(f"Successfully connected to Hypha server: {self.orchestrator_hypha_server_url}")

            service_api = {
                "name": "Orchestrator Manager (Simulation)",
                "id": self.orchestrator_hypha_service_id,
                "config": {
                    "visibility": "public", 
                    "run_in_executor": True,
                },
                "hello_orchestrator": self.hello_orchestrator,
                "add_imaging_task": self.add_imaging_task,
                "delete_imaging_task": self.delete_imaging_task,
                "get_all_imaging_tasks": self.get_all_imaging_tasks,
            }
            
            registered_service = await self.orchestrator_hypha_server_connection.register_service(service_api)
            logger.info(f"Orchestrator management service registered successfully. Service ID: {registered_service.id}")
            
            try: # Log proxy URL for easy testing
                ws = self.orchestrator_hypha_server_connection.config.workspace or "public" # Adjust if workspace is known
                sid_full = registered_service.id
                if ":" in sid_full:
                    sid = sid_full.split(":")[-1]
                else: # if id is already the short form
                    sid = sid_full

                proxy_url_base = self.orchestrator_hypha_server_url.replace("ws://", "http://").replace("wss://", "https://")
                logger.info(f"Test with hello_orchestrator: {proxy_url_base}/{ws}/services/{self.orchestrator_hypha_service_id}/hello_orchestrator or {proxy_url_base}/{ws}/services/{sid}/hello_orchestrator")
            except Exception as e:
                logger.debug(f"Could not construct proxy URL for testing: {e}")

        except Exception as e:
            logger.error(f"Failed to register orchestrator as a Hypha service: {e}")
            # Depending on criticality, might re-raise or set a flag

    @schema_function(skip_self=True)
    async def hello_orchestrator(self):
        """Returns a hello message from the orchestrator."""
        logger.info("hello_orchestrator service method called.")
        return "Hello from the Simulated Orchestrator!"

    @schema_function(skip_self=True)
    async def add_imaging_task(self, task_definition: dict):
        """Adds a new imaging task to config_simulation.json or updates it if name exists."""
        logger.info(f"Attempting to add/update imaging task: {task_definition.get('name')}")
        if not isinstance(task_definition, dict) or "name" not in task_definition or "settings" not in task_definition:
            msg = "Invalid task definition: must be a dict with 'name' and 'settings'."
            logger.error(msg)
            return {"success": False, "message": msg}

        task_name = task_definition["name"]
        new_settings = task_definition["settings"]

        required_settings = ["incubator_slot", "allocated_microscope", "pending_time_points", "imaging_zone", "Nx", "Ny", "illuminate_channels", "do_reflection_af"]
        for req_field in required_settings:
            if req_field not in new_settings:
                msg = f"Missing required field '{req_field}' in settings for task '{task_name}'."
                logger.error(msg)
                return {"success": False, "message": msg}
        
        # Validate pending_time_points format and content
        if not isinstance(new_settings["pending_time_points"], list):
            msg = f"'pending_time_points' must be a list for task '{task_name}'."
            logger.error(msg)
            return {"success": False, "message": msg}

        parsed_pending_time_points = []
        if not new_settings["pending_time_points"]: # Empty list is acceptable, means task is defined but has no work yet
            logger.warning(f"Task '{task_name}' has an empty 'pending_time_points' list.")
        
        for tp_str in new_settings["pending_time_points"]:
            try:
                # Basic ISO format validation for strings
                datetime.fromisoformat(tp_str.replace('Z', ''))
                parsed_pending_time_points.append(tp_str) # Keep as string for now, will be parsed by _load_and_update_tasks
            except ValueError as ve:
                msg = f"Invalid ISO format for a time point in 'pending_time_points' for task '{task_name}': {tp_str} ({ve})"
                logger.error(msg)
                return {"success": False, "message": msg}
        
        # Ensure imaged_time_points exists and is a list, even if empty, if not provided.
        if "imaged_time_points" not in new_settings:
            new_settings["imaged_time_points"] = []
        elif not isinstance(new_settings["imaged_time_points"], list):
            msg = f"'imaged_time_points' must be a list if provided for task '{task_name}'."
            logger.error(msg)
            return {"success": False, "message": msg}
        
        # Ensure imaging_started and imaging_completed flags exist, default to false
        if "imaging_started" not in new_settings:
            new_settings["imaging_started"] = False
        if "imaging_completed" not in new_settings:
            new_settings["imaging_completed"] = False


        async with self._config_lock:
            try:
                config_data = {"samples": []}
                try:
                    with open(CONFIG_FILE_PATH, 'r') as f:
                        config_data = json.load(f)
                except FileNotFoundError:
                    logger.warning(f"{CONFIG_FILE_PATH} not found. Will create a new one.")
                except json.JSONDecodeError:
                    logger.warning(f"{CONFIG_FILE_PATH} is corrupted. Will create a new one.")
                
                if "samples" not in config_data or not isinstance(config_data["samples"], list):
                     config_data["samples"] = []

                task_exists_at_index = -1
                for i, existing_task in enumerate(config_data["samples"]):
                    if existing_task.get("name") == task_name:
                        task_exists_at_index = i
                        break
                
                # Determine next_run_time_utc from the earliest pending time point
                next_run_from_pending = None
                if parsed_pending_time_points:
                    # Sort string time points to find the earliest one
                    sorted_pending_tp_strings = sorted(parsed_pending_time_points)
                    next_run_from_pending = sorted_pending_tp_strings[0]

                op_state = {
                    "status": "pending",
                    # Use the earliest pending time point or None if list is empty
                    "next_run_time_utc": next_run_from_pending if next_run_from_pending else datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    "retries": 0,
                    "last_updated_by_orchestrator": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                }
                if not next_run_from_pending:
                    op_state["status"] = "idle_no_pending_points" # Or "completed" if appropriate from start
                    if not new_settings.get("imaged_time_points"): # if no imaged points either
                        op_state["status"] = "completed" # Task is added with no work.
                        new_settings["imaging_completed"] = True # Reflect this in settings
                    else: # Has imaged points, but no pending ones
                        op_state["status"] = "completed"
                        new_settings["imaging_started"] = True
                        new_settings["imaging_completed"] = True


                if task_exists_at_index != -1:
                    logger.info(f"Task '{task_name}' already exists. Updating its settings and operational_state.")
                    config_data["samples"][task_exists_at_index]["settings"] = new_settings
                    config_data["samples"][task_exists_at_index]["operational_state"] = op_state
                else:
                    logger.info(f"Adding new task '{task_name}'.")
                    new_task_entry = {
                        "name": task_name,
                        "settings": new_settings,
                        "operational_state": op_state
                    }
                    config_data["samples"].append(new_task_entry)

                with open(CONFIG_FILE_PATH, 'w') as f:
                    json.dump(config_data, f, indent=4)
                logger.info(f"Task '{task_name}' processed (added/updated) in {CONFIG_FILE_PATH}.")

            except Exception as e:
                logger.error(f"Failed to add/update imaging task '{task_name}' in config: {e}", exc_info=True)
                return {"success": False, "message": f"Error processing task: {str(e)}"}

        await self._load_and_update_tasks() # Refresh orchestrator's internal task list
        return {"success": True, "message": f"Task '{task_name}' added/updated successfully."}

    @schema_function(skip_self=True)
    async def delete_imaging_task(self, task_name: str):
        """Deletes an imaging task from the simulation configuration."""
        logger.info(f"Attempting to delete imaging task: {task_name}")
        if not task_name:
            return {"success": False, "message": "Task name cannot be empty."}

        async with self._config_lock:
            try:
                config_data = {"samples": []}
                try:
                    with open(CONFIG_FILE_PATH, 'r') as f:
                        config_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    logger.warning(f"{CONFIG_FILE_PATH} not found or corrupted. Cannot delete task.")
                    return {"success": False, "message": f"{CONFIG_FILE_PATH} not found or corrupted."}

                if "samples" not in config_data or not isinstance(config_data["samples"], list):
                    logger.warning(f"No 'samples' list in {CONFIG_FILE_PATH}. Cannot delete task.")
                    return {"success": False, "message": "No 'samples' list in configuration."}

                original_count = len(config_data["samples"])
                config_data["samples"] = [task for task in config_data["samples"] if task.get("name") != task_name]
                
                if len(config_data["samples"]) == original_count:
                    logger.warning(f"Task '{task_name}' not found in {CONFIG_FILE_PATH}. No deletion occurred.")
                    return {"success": False, "message": f"Task '{task_name}' not found."}

                with open(CONFIG_FILE_PATH, 'w') as f:
                    json.dump(config_data, f, indent=4)
                logger.info(f"Task '{task_name}' deleted from {CONFIG_FILE_PATH}.")

            except Exception as e:
                logger.error(f"Failed to delete imaging task '{task_name}' from config: {e}", exc_info=True)
                return {"success": False, "message": f"Error deleting task: {str(e)}"}
        
        await self._load_and_update_tasks() # Refresh orchestrator's internal task list
        return {"success": True, "message": f"Task '{task_name}' deleted successfully."}

    @schema_function(skip_self=True)
    async def get_all_imaging_tasks(self):
        """Retrieves all imaging task configurations from config_simulation.json."""
        logger.debug(f"Attempting to read all imaging tasks from {CONFIG_FILE_PATH}")
        # Using a read lock isn't strictly necessary if writes are well-protected,
        # but can prevent reading a partially written file if a write operation was huge and slow (not the case here).
        async with self._config_lock:
            try:
                with open(CONFIG_FILE_PATH, 'r') as f:
                    config_data = json.load(f)
                return config_data.get("samples", []) # Return the list of samples
            except FileNotFoundError:
                logger.warning(f"{CONFIG_FILE_PATH} not found when trying to get all tasks.")
                return [] 
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {CONFIG_FILE_PATH} when getting all tasks.")
                return {"error": "Failed to decode configuration file.", "success": False}
            except Exception as e:
                logger.error(f"Failed to get all imaging tasks: {e}", exc_info=True)
                return {"error": str(e), "success": False}

    async def _start_health_check(self, service_type, service_instance):
        if service_type in self.health_check_tasks and not self.health_check_tasks[service_type].done():
            logger.info(f"Health check for {service_type} (sim) already running.")
            return
        logger.info(f"Starting health check for {service_type} (sim)...")
        # Pass the actual service instance and the type string
        task = asyncio.create_task(self.check_service_health(service_instance, service_type))
        self.health_check_tasks[service_type] = task

    async def _stop_health_check(self, service_type):
        if service_type in self.health_check_tasks:
            task = self.health_check_tasks.pop(service_type)
            if task and not task.done():
                logger.info(f"Stopping health check for {service_type} (sim)...")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Health check for {service_type} (sim) cancelled.")

    async def _load_and_update_tasks(self):
        logger.info(f"Loading and updating tasks from {CONFIG_FILE_PATH} (sim)")
        new_task_configs = {}
        raw_config_data = None # To store the full structure for writing back

        async with self._config_lock:
            try:
                with open(CONFIG_FILE_PATH, 'r') as f:
                    raw_config_data = json.load(f) # Load the raw structure
            except FileNotFoundError:
                logger.error(f"Configuration file {CONFIG_FILE_PATH} not found for simulation.")
                raw_config_data = {"samples": []} # Create a default structure if not found
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {CONFIG_FILE_PATH} for simulation. Will not update tasks from file this cycle.")
                return # Don't proceed if config is corrupt

        # Reset operational_state for all samples in the loaded raw_config_data is NO LONGER DONE HERE.
        # Operational state should be respected unless specifically reset by an API call or internal logic.

        current_time_utc = datetime.now(timezone.utc)

        for sample_config_from_file in raw_config_data.get("samples", []):
            task_name = sample_config_from_file.get("name")
            settings = sample_config_from_file.get("settings")

            if not task_name or not settings:
                logger.warning(f"Found a sample configuration without a name or settings in {CONFIG_FILE_PATH}. Skipping: {sample_config_from_file}")
                continue

            try:
                # Parse pending_time_points and imaged_time_points
                pending_datetimes = []
                for tp_str in settings.get("pending_time_points", []):
                    dt_obj = datetime.fromisoformat(tp_str.replace('Z', '') + '+00:00') if tp_str.endswith('Z') else datetime.fromisoformat(tp_str).replace(tzinfo=timezone.utc)
                    pending_datetimes.append(dt_obj)
                pending_datetimes.sort() # Ensure they are in chronological order

                imaged_datetimes = []
                for tp_str in settings.get("imaged_time_points", []):
                    dt_obj = datetime.fromisoformat(tp_str.replace('Z', '') + '+00:00') if tp_str.endswith('Z') else datetime.fromisoformat(tp_str).replace(tzinfo=timezone.utc)
                    imaged_datetimes.append(dt_obj)
                imaged_datetimes.sort()
                
                # Update imaging_started and imaging_completed flags based on parsed time points
                # These flags in settings are now more like status indicators derived from time points
                settings["imaging_started"] = bool(imaged_datetimes or (pending_datetimes and min(pending_datetimes) < current_time_utc) )
                settings["imaging_completed"] = not pending_datetimes and bool(imaged_datetimes) # Completed if no pending and some imaged
                if not pending_datetimes and not imaged_datetimes: # No work defined at all
                    settings["imaging_started"] = False
                    settings["imaging_completed"] = True 

                parsed_settings_config = {
                    "name": task_name,
                    "incubator_slot": settings["incubator_slot"],
                    "allocated_microscope": settings.get("allocated_microscope", SIM_DEFAULT_MICROSCOPE_ID),
                    "imaging_zone": settings["imaging_zone"],
                    "Nx": settings["Nx"],
                    "Ny": settings["Ny"],
                    "illuminate_channels": settings["illuminate_channels"],
                    "do_reflection_af": settings["do_reflection_af"],
                    # Store the parsed datetime objects for internal use
                    "pending_datetimes": pending_datetimes, 
                    "imaged_datetimes": imaged_datetimes,
                    # Also keep the flags for easier access if needed, though they can be derived
                    "imaging_started_flag": settings["imaging_started"],
                    "imaging_completed_flag": settings["imaging_completed"]
                }
                new_task_configs[task_name] = parsed_settings_config
            except KeyError as e:
                logger.error(f"Missing key {e} in simulation configuration settings for sample {task_name}. Skipping.")
                continue
            except ValueError as e:
                logger.error(f"Error parsing simulation configuration settings for sample {task_name}: {e}. Skipping.")
                continue
        
        tasks_to_remove = [name for name in self.tasks if name not in new_task_configs]
        for task_name in tasks_to_remove:
            logger.info(f"Task {task_name} removed from simulation configuration. Deactivating.")
            if self.active_task_name == task_name:
                logger.warning(f"Active task {task_name} was removed from sim config.")
                self.active_task_name = None 
            del self.tasks[task_name]

        a_task_state_changed_for_write = False
        for task_name, current_settings_config in new_task_configs.items():
            operational_state_from_file = {}
            for sample_in_file in raw_config_data.get("samples", []):
                if sample_in_file.get("name") == task_name:
                    operational_state_from_file = sample_in_file.get("operational_state", {})
                    break
            
            next_run_time_from_pending = current_settings_config["pending_datetimes"][0] if current_settings_config["pending_datetimes"] else None

            if task_name not in self.tasks:
                logger.info(f"New sim task added: {task_name}")
                persisted_status = operational_state_from_file.get("status", "pending")
                persisted_retries = operational_state_from_file.get("retries", 0)
                persisted_next_run_str = operational_state_from_file.get("next_run_time_utc")
                
                next_run_time_init = next_run_time_from_pending
                if persisted_next_run_str and not next_run_time_init: # No pending points, but had a persisted run time
                    try:
                        dt_val = datetime.fromisoformat(persisted_next_run_str.replace('Z', '') + '+00:00') if persisted_next_run_str.endswith('Z') else datetime.fromisoformat(persisted_next_run_str).replace(tzinfo=timezone.utc)
                        # If this persisted time makes sense (e.g., for retries or before all pending points were known), use it.
                        # However, if there are no pending_datetimes, next_run_time_init should remain None or be set to a far future time.
                        # For now, if next_run_time_from_pending is None, this persisted one is likely outdated or refers to a past state.
                        if not next_run_time_from_pending: # Only use if no pending points define a clear next run
                             logger.debug(f"Task '{task_name}': No pending points, but loaded persisted next_run_time_utc '{persisted_next_run_str}'. This might be stale.")
                             # next_run_time_init remains None if no pending points.
                        # If we decide to use it: next_run_time_init = dt_val 
                    except ValueError:
                        logger.warning(f"Task '{task_name}': Could not parse persisted next_run_time_utc '{persisted_next_run_str}'. Using earliest pending point or None.")
                elif next_run_time_init is None: # No pending points and no useful persisted run time string
                     logger.info(f"Task '{task_name}' has no pending time points. Will not schedule for run unless points are added.")
                     persisted_status = "idle_no_pending_points" if not current_settings_config["imaging_completed_flag"] else "completed"

                self.tasks[task_name] = {
                    "config": current_settings_config, 
                    "status": persisted_status,
                    "next_run_time": next_run_time_init, # This is now a datetime object or None
                    "retries": persisted_retries,
                    "_raw_settings_from_input": copy.deepcopy(sample_config_from_file.get("settings", {})) # Store raw settings
                }
                a_task_state_changed_for_write = True

                # If an initial next_run_time is set and is past, adjust it (only if task not completed/error)
                if self.tasks[task_name]["next_run_time"] and \
                   self.tasks[task_name]["next_run_time"] < current_time_utc and \
                   self.tasks[task_name]["status"] not in ["completed", "error_max_retries", "idle_no_pending_points"]:
                    logger.debug(f"New/Loaded task {task_name} (status {self.tasks[task_name]['status']}) earliest pending time {self.tasks[task_name]['next_run_time']} is past. Setting next_run_time to current_time_utc {current_time_utc}")
                    self.tasks[task_name]["next_run_time"] = current_time_utc
            else:
                # Task exists, update its 'config' part if changed
                # Deep comparison might be needed if lists of datetimes are involved in config
                if self.tasks[task_name]["config"]["pending_datetimes"] != current_settings_config["pending_datetimes"] or \
                   self.tasks[task_name]["config"]["imaged_datetimes"] != current_settings_config["imaged_datetimes"] or \
                   any(self.tasks[task_name]["config"].get(k) != current_settings_config.get(k) for k in ["incubator_slot", "allocated_microscope", "imaging_zone", "Nx", "Ny"]):
                    logger.info(f"Configuration settings or time points for sim task {task_name} updated.")
                    self.tasks[task_name]["config"] = current_settings_config
                    # If pending points changed, recalculate next_run_time
                    new_next_run_from_pending = current_settings_config["pending_datetimes"][0] if current_settings_config["pending_datetimes"] else None
                    if self.tasks[task_name]["next_run_time"] != new_next_run_from_pending:
                        logger.info(f"Task '{task_name}' next_run_time updated due to config change from {self.tasks[task_name]['next_run_time']} to {new_next_run_from_pending}")
                        self.tasks[task_name]["next_run_time"] = new_next_run_from_pending
                        # If this new next_run_time is past, adjust to current_time_utc if task is active
                        if new_next_run_from_pending and new_next_run_from_pending < current_time_utc and self.tasks[task_name]["status"] not in ["completed", "error_max_retries", "idle_no_pending_points"]:
                            self.tasks[task_name]["next_run_time"] = current_time_utc
                            logger.info(f"Task '{task_name}' new next_run_time was past, adjusted to current UTC.")

                    a_task_state_changed_for_write = True 

                self.tasks[task_name]["_raw_settings_from_input"] = copy.deepcopy(sample_config_from_file.get("settings", {}))

            # Adjust status if no pending points and task was not already completed/error
            task_state_dict = self.tasks[task_name]
            if not task_state_dict["config"]["pending_datetimes"] and task_state_dict["status"] not in ["completed", "error_max_retries"]:
                new_status = "completed" if task_state_dict["config"]["imaged_datetimes"] else "idle_no_pending_points" # if imaged some, it's done. If not, it's idle.
                if task_state_dict["status"] != new_status:
                    logger.info(f"Task '{task_name}' has no pending time points. Updating status from '{task_state_dict['status']}' to '{new_status}'.")
                    task_state_dict["status"] = new_status
                    task_state_dict["next_run_time"] = None # No more runs
                    a_task_state_changed_for_write = True
            
            # If a task is pending/error and its next_run_time (from earliest pending) is past, make it run now.
            current_task_status = task_state_dict["status"]
            task_next_run = task_state_dict["next_run_time"]
            if task_next_run and task_next_run <= current_time_utc and current_task_status not in ["completed", "error_max_retries", "active", "waiting_for_next_run", "idle_no_pending_points"]:
                 if task_state_dict["next_run_time"] != current_time_utc: # Avoid redundant logging/writes
                    logger.info(f"Task '{task_name}' (status: {current_task_status}) re-evaluated by load_tasks. Earliest pending time {task_next_run.isoformat()} is past or now. Overriding next_run_time to current_time_utc {current_time_utc.isoformat()}.")
                    task_state_dict["next_run_time"] = current_time_utc
                    a_task_state_changed_for_write = True

        if a_task_state_changed_for_write or tasks_to_remove:
            await self._write_tasks_to_config()

    async def _write_tasks_to_config(self):
        """Writes the current state of all tasks back to the configuration file."""
        logger.debug(f"Attempting to write tasks state to {CONFIG_FILE_PATH}")
        
        output_config_data = {"samples": []}
        
        async with self._config_lock: 
            try:
                with open(CONFIG_FILE_PATH, 'r') as f_read:
                    existing_data = json.load(f_read)
                    for key, value in existing_data.items():
                        if key != "samples":
                            output_config_data[key] = value
            except (FileNotFoundError, json.JSONDecodeError):
                 logger.warning(f"Could not re-read {CONFIG_FILE_PATH} before writing, or it was missing/corrupt. Will create/overwrite with current task data only.")

            for task_name, task_data_internal in self.tasks.items():
                settings_to_write = copy.deepcopy(task_data_internal.get("_raw_settings_from_input", {}))
                current_internal_config = task_data_internal["config"]

                # Update pending_time_points and imaged_time_points in settings_to_write from internal datetimes
                settings_to_write["pending_time_points"] = sorted([
                    dt.strftime('%Y-%m-%dT%H:%M:%SZ') for dt in current_internal_config.get("pending_datetimes", [])
                ])
                settings_to_write["imaged_time_points"] = sorted([
                    dt.strftime('%Y-%m-%dT%H:%M:%SZ') for dt in current_internal_config.get("imaged_datetimes", [])
                ])

                # Update imaging_started and imaging_completed flags based on the current state of time points
                has_pending = bool(current_internal_config.get("pending_datetimes"))
                has_imaged = bool(current_internal_config.get("imaged_datetimes"))
                
                settings_to_write["imaging_started"] = has_imaged or (has_pending and task_data_internal["status"] != "pending") # Considered started if any TP imaged, or if first TP is being processed
                if not has_pending and not has_imaged: # No points defined at all
                    settings_to_write["imaging_started"] = False 
                    settings_to_write["imaging_completed"] = True
                elif not has_pending and has_imaged: # All pending are done, some were imaged
                    settings_to_write["imaging_started"] = True
                    settings_to_write["imaging_completed"] = True
                else: # Still pending points, or pending but none imaged yet
                    settings_to_write["imaging_completed"] = False
                    if not has_imaged and task_data_internal["status"] == "pending": # Not yet started imaging the first point
                         settings_to_write["imaging_started"] = False


                # Operational state serialization
                next_run_time_str = task_data_internal["next_run_time"].strftime('%Y-%m-%dT%H:%M:%SZ') if task_data_internal.get("next_run_time") else None
                
                sample_entry = {
                    "name": task_name,
                    "settings": settings_to_write, 
                    "operational_state": {
                        "status": task_data_internal["status"],
                        "next_run_time_utc": next_run_time_str, 
                        "retries": task_data_internal["retries"],
                        "last_updated_by_orchestrator": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                    }
                }
                output_config_data["samples"].append(sample_entry)
            
            try:
                with open(CONFIG_FILE_PATH, 'w') as f_write:
                    json.dump(output_config_data, f_write, indent=4)
                logger.info(f"Successfully wrote tasks state to {CONFIG_FILE_PATH}")
            except IOError as e:
                logger.error(f"Error writing tasks state to {CONFIG_FILE_PATH}: {e}")
                
    async def _update_task_state_and_write_config(self, task_name, status=None, next_run_time=None, increment_retries=False, current_tp_to_move_to_imaged: datetime = None):
        """Helper to update task state (including time points) and write to config."""
        if task_name not in self.tasks:
            logger.warning(f"_update_task_state_and_write_config: Task {task_name} not found.")
            return

        changed = False
        task_state = self.tasks[task_name]
        task_config_internal = task_state["config"] # This holds pending_datetimes, imaged_datetimes

        if status and task_state["status"] != status:
            logger.info(f"Task '{task_name}' status changing from '{task_state['status']}' to '{status}'")
            task_state["status"] = status
            changed = True
        
        # Handle moving a time point from pending to imaged
        if current_tp_to_move_to_imaged:
            if current_tp_to_move_to_imaged in task_config_internal["pending_datetimes"]:
                task_config_internal["pending_datetimes"].remove(current_tp_to_move_to_imaged)
                task_config_internal["imaged_datetimes"].append(current_tp_to_move_to_imaged)
                task_config_internal["imaged_datetimes"].sort() # Keep it sorted
                logger.info(f"Moved time point {current_tp_to_move_to_imaged.isoformat()} to imaged for task '{task_name}'.")
                changed = True
                
                # After moving, if pending_datetimes is now empty, the task might be completed.
                if not task_config_internal["pending_datetimes"]:
                    logger.info(f"Task '{task_name}' has no more pending time points after imaging {current_tp_to_move_to_imaged.isoformat()}.")
                    if task_state["status"] != "completed":
                        logger.info(f"Marking task '{task_name}' as completed.")
                        task_state["status"] = "completed"
                    next_run_time = None # No more runs for this task
                # else, next_run_time will be updated below from the new earliest pending point
            else:
                logger.warning(f"Time point {current_tp_to_move_to_imaged.isoformat()} not found in pending_datetimes for task '{task_name}'. Cannot move.")

        # If status is changing to something that implies a run (e.g. waiting_for_next_run) or TP moved,
        # or if next_run_time is explicitly provided, update it.
        # The new next_run_time should be the earliest of the remaining pending_datetimes.
        if next_run_time is not Ellipsis: # Use Ellipsis to signal explicit None vs. recalculate
            if current_tp_to_move_to_imaged and not task_config_internal["pending_datetimes"]:
                # This case is handled above, task becomes completed, next_run_time is None
                pass 
            elif next_run_time is not None: # Explicitly set next_run_time (e.g. for retries)
                if task_state["next_run_time"] != next_run_time:
                    logger.info(f"Task '{task_name}' next_run_time explicitly changing from '{task_state.get('next_run_time')}' to '{next_run_time.isoformat() if next_run_time else 'None'}'")
                    task_state["next_run_time"] = next_run_time
            elif task_config_internal["pending_datetimes"]: # Recalculate from pending points
                new_next_run_from_pending = task_config_internal["pending_datetimes"][0]
                if task_state["next_run_time"] != new_next_run_from_pending:
                    logger.info(f"Task '{task_name}' next_run_time recalculated to '{new_next_run_from_pending.isoformat()}' from pending points.")
                    task_state["next_run_time"] = new_next_run_from_pending
            elif not task_config_internal["pending_datetimes"] and task_state["status"] != "completed":
                 # No pending points, not completed (e.g. error, idle). Set next_run_time to None if it wasn't already.
                 if task_state["next_run_time"] is not None:
                    logger.info(f"Task '{task_name}' has no pending points and status is {task_state['status']}. Setting next_run_time to None.")
                    task_state["next_run_time"] = None
                    changed = True
            # if next_run_time is None (explicitly passed) and task status is not completed, it will be set.
            # This also covers the case where task became completed and next_run_time was set to None above.
            elif next_run_time is None and task_state["next_run_time"] is not None: # Explicitly clearing next_run_time
                 logger.info(f"Task '{task_name}' next_run_time explicitly set to None.")
                 task_state["next_run_time"] = None
                 changed = True

        if increment_retries:
            task_state["retries"] += 1
            logger.info(f"Task '{task_name}' retries incremented to {task_state['retries']}")
            changed = True
        elif "retries" not in task_state : # ensure retries is initialized if not present
             task_state["retries"] = 0

        if changed:
            await self._write_tasks_to_config()

    async def setup_connections(self, target_microscope_id_from_task=None):
        logger.info(f"Sim: setup_connections called for target microscope: {target_microscope_id_from_task}")
        # No actual Hypha connections, just ensure mock objects are ready
        # and health checks are (re)started.

        if not self.incubator: # Should always be true due to __init__
            self.incubator = MockIncubator(self.sim_incubator_id, asyncio.get_event_loop())
            logger.info(f"Simulated Incubator ({self.sim_incubator_id}) ensured/re-initialized.")
        await self._start_health_check('incubator', self.incubator)
            
        if not self.robotic_arm: # Should always be true
            self.robotic_arm = MockRoboticArm(self.sim_robotic_arm_id, asyncio.get_event_loop())
            logger.info(f"Simulated Robotic Arm ({self.sim_robotic_arm_id}) ensured/re-initialized.")
        await self._start_health_check('robotic_arm', self.robotic_arm)

        if target_microscope_id_from_task:
            if self.current_microscope_id != target_microscope_id_from_task or not self.current_mock_microscope_instance:
                if self.current_mock_microscope_instance: # Connected to a different logical microscope
                    logger.info(f"Sim: Switching mock microscope from {self.current_microscope_id} to {target_microscope_id_from_task}")
                    await self.disconnect_single_service('microscope') # Stops health check and clears instance
                
                logger.info(f"Sim: Initializing mock microscope for {target_microscope_id_from_task}...")
                # In simulation, the backend service ID might be the same, but we create a new mock instance
                # to simulate a fresh connection or represent a different logical device.
                self.current_mock_microscope_instance = MockMicroscope(target_microscope_id_from_task, asyncio.get_event_loop())
                self.microscope = self.current_mock_microscope_instance # Assign to self.microscope for generic calls
                self.current_microscope_id = target_microscope_id_from_task
                logger.info(f"Simulated Microscope '{self.current_microscope_id}' initialized and active.")
                await self._start_health_check('microscope', self.microscope)
            else:
                logger.info(f"Simulated Microscope '{target_microscope_id_from_task}' already active.")
                # Ensure health check is running if it somehow stopped
                if 'microscope' not in self.health_check_tasks or self.health_check_tasks['microscope'].done():
                   if self.microscope: # Should be self.current_mock_microscope_instance
                       await self._start_health_check('microscope', self.microscope)
        else: 
            if self.current_mock_microscope_instance:
                logger.info(f"No target sim microscope specified. Disconnecting current: {self.current_microscope_id}.")
                await self.disconnect_single_service('microscope')
        
        # Return true if essential mock services seem to be there.
        # The 'microscope' part is more about whether the *intended* mock is set.
        return bool(self.incubator and self.robotic_arm)

    async def check_service_health(self, service, service_type_str):
        """Check if the service is healthy and reset if needed (simulated)"""
        service_name = service.id if hasattr(service, "id") else f"simulated_{service_type_str}_service"
            
        while True:
            try:
                task_statuses = await service.get_all_task_status()
                if any(status == "failed" for status in task_statuses.values()):
                    logger.error(f"{service_name} (sim) has failed tasks: {task_statuses}")
                    logger.warning("Simulated service has failed tasks. Attempting reset.")
                    raise Exception("Service not healthy (simulated task failure)")

                hello_world_result = await service.hello_world()
                if hello_world_result != "Hello world":
                    logger.error(f"{service_name} (sim) hello_world check failed: {hello_world_result}")
                    raise Exception("Simulated service not healthy (hello_world failed)")
                logger.debug(f"Health check passed for {service_name} (sim)")
            except Exception as e:
                logger.error(f"{service_name} (sim) health check failed: {e}")
                logger.info(f"Attempting to reset simulated {service_type_str} service...")
                await self.disconnect_single_service(service_type_str)
                await self.reconnect_single_service(service_type_str) # Pass the type string
            await asyncio.sleep(30)

    async def disconnect_single_service(self, service_type):
        """Disconnect a specific simulated service."""
        await self._stop_health_check(service_type)
        # No actual disconnect calls to Hypha, just clear our references to mocks
        if service_type == 'incubator' and self.incubator:
            logger.info(f"Sim: Clearing incubator mock instance.")
            # await self.incubator.disconnect() # Mock disconnect if it does something useful
            self.incubator = None # Or re-init with a fresh mock if preferred on reconnect
        elif service_type == 'microscope' and self.current_mock_microscope_instance:
            logger.info(f"Sim: Clearing microscope mock instance ({self.current_microscope_id}).")
            # await self.current_mock_microscope_instance.disconnect()
            self.current_mock_microscope_instance = None
            self.microscope = None 
            self.current_microscope_id = None
        elif service_type == 'robotic_arm' and self.robotic_arm:
            logger.info(f"Sim: Clearing robotic_arm mock instance.")
            # await self.robotic_arm.disconnect()
            self.robotic_arm = None
        logger.info(f"Simulated {service_type} service instance cleared/mock-disconnected.")
    
    async def reconnect_single_service(self, service_type):
        """Reconnect a specific simulated service by re-initializing its mock."""
        logger.info(f"Sim: Reconnecting/Re-initializing mock for {service_type} service...")
        loop = asyncio.get_event_loop()
        if service_type == 'incubator':
            self.incubator = MockIncubator(self.sim_incubator_id, loop)
            logger.info(f"Simulated Incubator re-initialized.")
            await self._start_health_check('incubator', self.incubator)
        elif service_type == 'microscope':
            # When reconnecting a microscope, use the current_microscope_id if available,
            # otherwise, it implies a general microscope service issue, might use a default.
            target_id = self.current_microscope_id or SIM_DEFAULT_MICROSCOPE_ID # Fallback if no specific task was active
            self.current_mock_microscope_instance = MockMicroscope(target_id, loop)
            self.microscope = self.current_mock_microscope_instance
            self.current_microscope_id = target_id # Ensure current_microscope_id is set
            logger.info(f"Simulated Microscope ({self.current_microscope_id}) re-initialized.")
            await self._start_health_check('microscope', self.microscope)
        elif service_type == 'robotic_arm':
            self.robotic_arm = MockRoboticArm(self.sim_robotic_arm_id, loop)
            logger.info(f"Simulated Robotic Arm re-initialized.")
            await self._start_health_check('robotic_arm', self.robotic_arm)
        else:
            logger.error(f"Sim: Unknown service type '{service_type}' for reconnect.")

    async def disconnect_services(self):
        logger.info("Disconnecting all simulated services...")
        service_types = []
        if self.incubator: service_types.append('incubator')
        if self.microscope: service_types.append('microscope')
        if self.robotic_arm: service_types.append('robotic_arm')
        
        for stype in service_types:
            await self.disconnect_single_service(stype)
        logger.info("Disconnect process completed for all simulated services.")

    async def call_service_with_retries(self, service_type, method_name, *args, max_retries=3, timeout=10, **kwargs): # Shorter retries/timeout for sim
        retries = 0
        while retries < max_retries:
            service = None
            if service_type == 'incubator': service = self.incubator
            elif service_type == 'microscope': service = self.microscope
            elif service_type == 'robotic_arm': service = self.robotic_arm
            
            if not service:
                logger.error(f"Sim Service {service_type} is not available. Retrying... ({retries + 1}/{max_retries})")
                retries += 1
                await asyncio.sleep(timeout / 2) 
                continue
            try:
                status = await service.get_task_status(method_name)
                logger.info(f"Sim Task {method_name} status: {status}")
                if status == "failed":
                    logger.error(f"Sim Task {method_name} failed. Stopping execution for this call.")
                    await service.reset_task_status(method_name) # Reset for next sim attempt
                    return False

                if status == "not_started":
                    logger.info(f"Sim Starting the task {method_name} by calling mock...")
                    try:
                        # The mock method itself handles setting status to "running" and then "finished"/"failed"
                        # and simulates duration.
                        method_to_call = getattr(service, method_name)
                        await asyncio.wait_for(method_to_call(*args, **kwargs), timeout=timeout + 1) # Ensure mock's internal sleep is less than this
                        
                        # After the call, check status again (it should be finished or failed)
                        status = await service.get_task_status(method_name) 
                        logger.info(f"Sim Task {method_name} status after mock call: {status}")

                    except asyncio.TimeoutError:
                        logger.warning(f"Sim Operation {method_name} (mock call) timed out. Mock task status: {await service.get_task_status(method_name)}")
                        self._tasks_status[method_name] = "failed" # Force fail on timeout for sim
                        status = "failed" 

                if status == "running": # If mock is more complex and needs polling after start
                    polling_attempts = 0
                    max_polling_attempts = timeout # Poll for roughly the timeout duration
                    while polling_attempts < max_polling_attempts :
                        status = await service.get_task_status(method_name)
                        logger.info(f"Sim Polling Task {method_name} status: {status}")
                        if status == "finished" or status == "failed":
                            break
                        await asyncio.sleep(1) 
                        polling_attempts +=1
                    if status == "running": # Still running after polling
                        logger.warning(f"Sim Task {method_name} still 'running' after polling. Treating as failed for this attempt.")
                        status = "failed" # Force fail

                if status == "finished":
                    logger.info(f"Sim Task {method_name} completed successfully (as per mock).")
                    await service.reset_task_status(method_name) 
                    return True
                elif status == "failed":
                    logger.error(f"Sim Task {method_name} failed (as per mock).")
                    await service.reset_task_status(method_name) 
                    return False
                # else: status might be 'not_started' if something went wrong before mock was called, or unexpected status
                # This path should ideally not be hit if mock call was attempted.

            except Exception as e:
                logger.error(f"Sim Error in call_service_with_retries for {method_name}: {e}. Retrying... ({retries + 1}/{max_retries})")
            retries += 1
            await asyncio.sleep(timeout / 2) 
        logger.error(f"Sim Max retries reached for task {method_name}.")
        return False

    async def load_plate_from_incubator_to_microscope(self, incubator_slot): # Parameterized
        if self.sample_on_microscope_flag:
            logger.info("Sim: Sample plate already on microscope")
            return True
        logger.info(f"Sim: Loading sample from incubator slot {incubator_slot}...")
        p1 = self.call_service_with_retries('incubator', "get_sample_from_slot_to_transfer_station", incubator_slot, timeout=10)
        p2 = self.call_service_with_retries('microscope', "home_stage", timeout=5)
        gather = await asyncio.gather(p1, p2)
        if not all(gather): return False
        if not await self.call_service_with_retries('robotic_arm', "grab_sample_from_incubator", timeout=10): return False
        if not await self.call_service_with_retries('robotic_arm', "transport_from_incubator_to_microscope1", timeout=10): return False
        if not await self.call_service_with_retries('robotic_arm', "put_sample_on_microscope1", timeout=10): return False
        if not await self.call_service_with_retries('microscope', "return_stage", timeout=5): return False
        logger.info("Sim: Sample loaded onto microscope.")
        self.sample_on_microscope_flag = True
        return True

    async def unload_plate_from_microscope(self, incubator_slot): # Parameterized
        if not self.sample_on_microscope_flag:
            logger.info("Sim: Sample plate not on microscope")
            return True
        logger.info(f"Sim: Unloading sample to incubator slot {incubator_slot}...")
        if not await self.call_service_with_retries('microscope', "home_stage", timeout=5): return False
        if not await self.call_service_with_retries('robotic_arm', "grab_sample_from_microscope1", timeout=10): return False
        if not await self.call_service_with_retries('robotic_arm', "transport_from_microscope1_to_incubator", timeout=10): return False
        if not await self.call_service_with_retries('robotic_arm', "put_sample_on_incubator", timeout=10): return False
        p1 = self.call_service_with_retries('incubator', "put_sample_from_transfer_station_to_slot", incubator_slot, timeout=10)
        p2 = self.call_service_with_retries('microscope', "return_stage", timeout=5)
        gather = await asyncio.gather(p1, p2)
        if not all(gather): return False
        logger.info("Sim: Sample unloaded from microscope.")
        self.sample_on_microscope_flag = False
        return True

    async def run_cycle(self, task_config):
        task_name = task_config["name"]
        incubator_slot = task_config["incubator_slot"]
        action_id = f"SIM_{task_name.replace(' ', '_')}-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        logger.info(f"Sim: Starting cycle for task: {task_name} with action_id: {action_id}")

        if not self.incubator or not self.microscope or not self.robotic_arm:
            logger.error(f"Sim: Services not available for task {task_name}. Attempting reconnect.")
            # setup_connections should be called by run_time_lapse before run_cycle
            # This is an additional check.
            if not await self.setup_connections(target_microscope_id_from_task=task_config["allocated_microscope"]):
                 logger.error(f"Sim: Failed to re-establish service connections for task {task_name}. Cycle aborted.")
                 return False
            if not self.microscope:
                 logger.error(f"Sim: Microscope {task_config['allocated_microscope']} still not available. Cycle aborted.")
                 return False

        try: # Resetting service states for simulation
            logger.info(f"Sim: Resetting task statuses on services for task {task_name}...")
            if self.microscope: await self.microscope.reset_all_task_status()
            if self.incubator: await self.incubator.reset_all_task_status()
            if self.robotic_arm: await self.robotic_arm.reset_all_task_status()
        except Exception as e:
            logger.warning(f"Sim: Error resetting task statuses: {e}")

        self.sample_on_microscope_flag = False 

        if not await self.load_plate_from_incubator_to_microscope(incubator_slot=incubator_slot):
            logger.error(f"Sim: Failed to load sample for task {task_name}")
            return False
        
        # Using parameters from task_config for scan_well_plate_simulated
        # The simulated method itself might not use all of them, but we pass them for consistency.
        scan_successful = await self.call_service_with_retries(
            'microscope',
            "scan_well_plate_simulated", # Ensure this method exists on the simulated microscope service
            illuminate_channels=task_config["illuminate_channels"],
            do_reflection_af=task_config["do_reflection_af"],
            scanning_zone=task_config["imaging_zone"],
            Nx=task_config["Nx"],
            Ny=task_config["Ny"],
            action_ID=action_id, # Pass the generated action_id
            timeout=60 # Simulated scan timeout
        )

        if not scan_successful:
            logger.error(f"Sim: Microscope scanning failed for task {task_name}.")
            logger.info(f"Sim: Attempting to unload plate for task {task_name} after scan failure.")
            if not await self.unload_plate_from_microscope(incubator_slot=incubator_slot):
                logger.error(f"Sim: Failed to unload sample for task {task_name} after scan error.")
            return False

        if not await self.unload_plate_from_microscope(incubator_slot=incubator_slot):
            logger.error(f"Sim: Failed to unload sample for task {task_name} after successful scan.")
            return False
        
        logger.info(f"Sim: Cycle for task {task_name} (action_id: {action_id}) completed successfully.")
        return True

    async def run_time_lapse(self): # Adapted from orchestrator.py
        logger.info("Simulated Orchestrator run_time_lapse started.")
        last_config_read_time = 0

        while True:
            current_time_utc = datetime.now(timezone.utc)
            logger.debug(f"Sim: run_time_lapse loop. Current time: {current_time_utc.isoformat()}")

            if (asyncio.get_event_loop().time() - last_config_read_time) > CONFIG_READ_INTERVAL:
                await self._load_and_update_tasks()
                last_config_read_time = asyncio.get_event_loop().time()

            next_task_to_run = None
            earliest_next_run_dt_obj = None # Stores the actual datetime object for comparison
            current_pending_tp_for_next_task = None # Stores the specific TP to be imaged

            if not self.tasks:
                logger.debug("Sim: No tasks loaded yet.")
            
            for task_name, task_data in list(self.tasks.items()):
                internal_config = task_data["config"]
                status = task_data["status"]
                # next_run_time from task_data is already a datetime object or None
                task_next_run_dt_obj = task_data["next_run_time"] 

                logger.debug(f"Sim: Checking task '{task_name}': status='{status}', next_run_dt_obj='{task_next_run_dt_obj.isoformat() if task_next_run_dt_obj else 'None'}', pending_points_count={len(internal_config['pending_datetimes'])}")

                if status in ["completed", "error_max_retries", "idle_no_pending_points"]:
                    logger.debug(f"Sim: Task '{task_name}' skipped due to status: {status}")
                    continue
                
                if not task_next_run_dt_obj: # Should not happen if status isn't one of the above, but a safeguard
                    logger.warning(f"Sim: Task '{task_name}' has status '{status}' but no next_run_time. Skipping.")
                    continue
                
                # Check if this task is due based on its next_run_time (earliest pending time point)
                if current_time_utc >= task_next_run_dt_obj:
                    logger.debug(f"Sim: Task '{task_name}' is eligible (next_run: {task_next_run_dt_obj.isoformat()}). Current earliest_next_run: {earliest_next_run_dt_obj.isoformat() if earliest_next_run_dt_obj else 'None'}")
                    if earliest_next_run_dt_obj is None or task_next_run_dt_obj < earliest_next_run_dt_obj:
                        earliest_next_run_dt_obj = task_next_run_dt_obj
                        next_task_to_run = task_name
                        current_pending_tp_for_next_task = internal_config["pending_datetimes"][0] # This is the TP to image
                        logger.debug(f"Sim: Task '{task_name}' provisionally selected for TP: {current_pending_tp_for_next_task.isoformat()}.")
                    else:
                        logger.debug(f"Sim: Task '{task_name}' eligible but its next_run_time ({task_next_run_dt_obj.isoformat()}) is not earlier than current earliest ({earliest_next_run_dt_obj.isoformat() if earliest_next_run_dt_obj else 'N/A'}).")
                else:
                    logger.debug(f"Sim: Task '{task_name}' not due yet (next_run: {task_next_run_dt_obj.isoformat()}).")
            
            if next_task_to_run and current_pending_tp_for_next_task:
                self.active_task_name = next_task_to_run
                task_data = self.tasks[self.active_task_name]
                task_config_for_cycle = task_data["config"] # This contains the parsed settings and datetime lists
                
                logger.info(f"Sim: Selected task {self.active_task_name} for time point {current_pending_tp_for_next_task.isoformat()}. Current state: status='{task_data['status']}', next_run='{task_data['next_run_time'].isoformat() if task_data['next_run_time'] else 'None'}'")

                if not await self.setup_connections(target_microscope_id_from_task=task_config_for_cycle["allocated_microscope"]):
                    logger.error(f"Sim: Failed to setup connections for task {self.active_task_name}. Retrying later.")
                    await self._update_task_state_and_write_config(
                        self.active_task_name,
                        status="error_connection_failed",
                        next_run_time=current_time_utc + CYCLE_RETRY_DELAY # Keep current TP as next run, but delay
                    )
                    self.active_task_name = None
                    await asyncio.sleep(ORCHESTRATOR_LOOP_SLEEP)
                    continue
                
                if not self.microscope or self.current_microscope_id != task_config_for_cycle["allocated_microscope"]:
                    logger.error(f"Sim: Microscope {task_config_for_cycle['allocated_microscope']} not available for {self.active_task_name}. Retrying later.")
                    await self._update_task_state_and_write_config(
                        self.active_task_name,
                        status="error_microscope_unavailable",
                        next_run_time=current_time_utc + CYCLE_RETRY_DELAY # Keep current TP as next run, but delay
                    )
                    self.active_task_name = None
                    await asyncio.sleep(ORCHESTRATOR_LOOP_SLEEP)
                    continue

                logger.info(f"Sim: Starting cycle for task: {self.active_task_name}, time point: {current_pending_tp_for_next_task.isoformat()}")
                await self._update_task_state_and_write_config(self.active_task_name, status="active")
                
                # Pass the full internal config to run_cycle, it has all necessary fields like Nx, Ny etc.
                cycle_success = await self.run_cycle(task_config_for_cycle) 

                if cycle_success:
                    logger.info(f"Sim: Cycle for task {self.active_task_name}, time point {current_pending_tp_for_next_task.isoformat()} success.")
                    # Move current_pending_tp_for_next_task to imaged_datetimes
                    # _update_task_state_and_write_config will also update next_run_time to the new earliest pending (or None if complete)
                    # and set status to 'completed' if no more pending points.
                    self.tasks[self.active_task_name]["retries"] = 0 # Reset retries on success for this TP
                    await self._update_task_state_and_write_config(
                        self.active_task_name,
                        status="waiting_for_next_run", # This status will be overridden to 'completed' by the helper if no more TPs
                        current_tp_to_move_to_imaged=current_pending_tp_for_next_task,
                        next_run_time=Ellipsis # Signal to recalculate from remaining pending points
                    )
                else: # Cycle failed for the current time point
                    logger.error(f"Sim: Cycle for task {self.active_task_name}, time point {current_pending_tp_for_next_task.isoformat()} failed.")
                    
                    current_retries = self.tasks[self.active_task_name]["retries"]
                    if current_retries + 1 >= MAX_CYCLE_RETRIES: 
                        logger.error(f"Sim: Max retries ({MAX_CYCLE_RETRIES}) will be reached for {self.active_task_name} on time point {current_pending_tp_for_next_task.isoformat()}. Marking task error_max_retries.")
                        # Keep the failed TP in pending_datetimes, but mark task as error_max_retries.
                        # next_run_time will effectively be None for this task.
                        await self._update_task_state_and_write_config(
                            self.active_task_name,
                            status="error_max_retries",
                            increment_retries=True,
                            next_run_time=None # Task stops trying
                        )
                    else:
                        logger.info(f"Sim: Scheduling retry for {self.active_task_name} on time point {current_pending_tp_for_next_task.isoformat()} (current retries: {current_retries}).")
                        # The next_run_time for retry will be current_time + delay. The current_pending_tp_for_next_task remains the one to retry.
                        await self._update_task_state_and_write_config(
                            self.active_task_name,
                            status="error_cycle_failed",
                            next_run_time=current_time_utc + CYCLE_RETRY_DELAY, # Retry this specific TP later
                            increment_retries=True
                        )
                self.active_task_name = None
            else:
                if self.active_task_name: self.active_task_name = None
                min_wait_time = ORCHESTRATOR_LOOP_SLEEP
                await asyncio.sleep(min_wait_time)

async def main():
    parser = argparse.ArgumentParser(description='Run the Simulated Orchestration System.')
    parser.add_argument('--local', action='store_true', help='Run in local sim mode')
    args = parser.parse_args()

    log_file_name = f"orchestrator-simulation-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    global logger # Define logger globally for the simulation script
    logger = setup_logging(log_file=log_file_name)

    orchestrator = OrchestrationSystem(local=args.local)
    # await orchestrator.setup_connections() # setup_connections is now called within run_time_lapse per task
    try:
        await orchestrator._register_self_as_hypha_service() # Register orchestrator's own Hypha service
        await orchestrator.run_time_lapse() # Removed round_time
    except KeyboardInterrupt:
        logger.info("Simulated Orchestrator shutting down (KeyboardInterrupt)...")
    finally:
        logger.info("Simulated Orchestrator performing cleanup...")
        if orchestrator:
            if orchestrator.orchestrator_hypha_server_connection:
                try:
                    # Ideally, unregister service if Hypha API supports it easily, or just disconnect.
                    # server.unregister_service(self.orchestrator_hypha_service_id) # Pseudocode
                    await orchestrator.orchestrator_hypha_server_connection.disconnect()
                    logger.info("Disconnected from Hypha server for orchestrator's own service.")
                except Exception as e:
                    logger.error(f"Error disconnecting orchestrator's Hypha service: {e}")
            await orchestrator.disconnect_services() # This disconnects mock services
        logger.info("Simulated Orchestrator cleanup complete.")

if __name__ == '__main__':
    asyncio.run(main())
