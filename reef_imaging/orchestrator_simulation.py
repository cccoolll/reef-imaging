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
from datetime import datetime, timedelta
import argparse
import json
import random # For simulating occasional failures
import copy # Added import

# Set up logging
def setup_logging(log_file="orchestrator_simulation.log", max_bytes=20*1024*1024, backup_count=3):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Rotating file handler with limit
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
ORCHESTRATOR_LOOP_SLEEP = 5 # Seconds, faster for simulation

# Custom exceptions for better error handling
class ServiceConnectionError(Exception):
    """Raised when service connection fails"""
    pass

class ServiceOperationError(Exception):
    """Raised when service operation fails"""
    pass

class MicroscopeNotAvailableError(Exception):
    """Raised when microscope is not available"""
    pass

class TransportOperationError(Exception):
    """Raised when transport operation fails"""
    pass

# --- Mock Hypha Services ---
class MockHyphaService:
    def __init__(self, service_id, service_type, loop):
        self.id = service_id
        self.service_type = service_type
        self._loop = loop
        self._method_call_counts = {}
        logger.info(f"MockHyphaService {self.id} ({self.service_type}) initialized.")

    async def hello_world(self):
        logger.debug(f"Mock {self.id}: hello_world() called.")
        await asyncio.sleep(0.01) # Simulate network delay
        return "Hello world"

    async def disconnect(self):
        logger.info(f"MockHyphaService {self.id} ({self.service_type}): disconnect() called.")
        await asyncio.sleep(0.01)

    async def _simulate_method_call(self, method_name, duration=0.5, fail_sometimes=False, fail_rate=0.0):
        self._method_call_counts[method_name] = self._method_call_counts.get(method_name, 0) + 1
        logger.info(f"Mock {self.id}: Starting method {method_name} (Call #{self._method_call_counts[method_name]})")
        await asyncio.sleep(duration / 2) # Halfway through
        
        # Simulate occasional failures
        if fail_sometimes and random.random() < fail_rate:
            error_msg = f"Mock {self.id}: Method {method_name} simulated failure"
            logger.error(error_msg)
            raise ServiceOperationError(error_msg)
        
        logger.info(f"Mock {self.id}: Finishing method {method_name}")
        await asyncio.sleep(duration / 2) # Remaining duration

class MockIncubator(MockHyphaService):
    def __init__(self, service_id, loop):
        super().__init__(service_id, "incubator", loop)

    async def get_sample_from_slot_to_transfer_station(self, slot, **kwargs):
        await self._simulate_method_call("get_sample_from_slot_to_transfer_station", duration=1)

    async def put_sample_from_transfer_station_to_slot(self, slot, **kwargs):
        await self._simulate_method_call("put_sample_from_transfer_station_to_slot", duration=1)

class MockRoboticArm(MockHyphaService):
    def __init__(self, service_id, loop):
        super().__init__(service_id, "robotic_arm", loop)

    async def grab_sample_from_incubator(self, **kwargs):
        await self._simulate_method_call("grab_sample_from_incubator", duration=0.5)

    async def transport_from_incubator_to_microscope1(self, **kwargs):
        await self._simulate_method_call("transport_from_incubator_to_microscope1", duration=1)

    async def put_sample_on_microscope1(self, **kwargs):
        await self._simulate_method_call("put_sample_on_microscope1", duration=0.5)

    async def grab_sample_from_microscope1(self, **kwargs):
        await self._simulate_method_call("grab_sample_from_microscope1", duration=0.5)

    async def transport_from_microscope1_to_incubator(self, **kwargs):
        await self._simulate_method_call("transport_from_microscope1_to_incubator", duration=1)

    async def put_sample_on_incubator(self, **kwargs):
        await self._simulate_method_call("put_sample_on_incubator", duration=0.5)


class MockMicroscope(MockHyphaService):
    def __init__(self, service_id, loop):
        super().__init__(service_id, "microscope", loop)

    async def home_stage(self, **kwargs):
        await self._simulate_method_call("home_stage", duration=0.3)

    async def return_stage(self, **kwargs):
        await self._simulate_method_call("return_stage", duration=0.3)

    async def scan_well_plate_simulated(self, illuminate_channels, do_reflection_af, scanning_zone, Nx, Ny, action_ID, **kwargs):
        logger.info(f"Mock {self.id}: scan_well_plate_simulated called with action_ID: {action_ID}, channels: {illuminate_channels}, zone: {scanning_zone}, Nx: {Nx}, Ny: {Ny}, AF: {do_reflection_af}")
        # Simulate a longer scan, maybe with a chance of failure
        await self._simulate_method_call("scan_well_plate_simulated", duration=2, fail_sometimes=True, fail_rate=0.05) # 5% chance of scan failure

# --- End Mock Hypha Services ---

class OrchestrationSystem:
    def __init__(self):
        # Orchestrator's own Hypha service registration details (always production-like)
        self.orchestrator_hypha_server_url = "https://hypha.aicell.io"
        self.workspace = "reef-imaging" # Default workspace for aicell.io
        self.token_for_orchestrator_registration = os.environ.get("REEF_WORKSPACE_TOKEN")
        
        self.orchestrator_hypha_service_id = "orchestrator-manager-simulation"
        self.orchestrator_hypha_server_connection = None

        # Mock services (do not connect via Hypha in this simulation)
        # Their URLs/tokens like REEF_LOCAL_TOKEN would be relevant if they were real Hypha services
        # For now, their configuration is just their simulated IDs.
        loop = asyncio.get_event_loop()
        self.incubator = MockIncubator(SIM_INCUBATOR_ID, loop)
        self.microscope = None 
        self.robotic_arm = MockRoboticArm(SIM_ROBOTIC_ARM_ID, loop)
        
        self.sample_on_microscope_flag = False
        self.sim_incubator_id = SIM_INCUBATOR_ID 
        self.sim_robotic_arm_id = SIM_ROBOTIC_ARM_ID 
        self.current_microscope_id = None 
        self.current_mock_microscope_instance = None 
        self.tasks = {}
        self.health_check_tasks = {}
        self.active_task_name = None
        self._config_lock = asyncio.Lock() 

        # Transport Queue and Worker Task
        self.transport_queue = asyncio.Queue()
        self._transport_worker_task = None # Will be created in _register_self_as_hypha_service

    async def _transport_worker_loop(self):
        logger.info("Transport worker loop started.")
        while True:
            try:
                # Get a transport task from the queue
                # Task details could be a dict: e.g., {"action": "load", "incubator_slot": 1, "future": future_obj}
                task_details = await self.transport_queue.get()
                logger.info(f"Transport worker picked up task: {task_details.get('action')} for slot {task_details.get('incubator_slot')}")
                
                action = task_details.get("action")
                incubator_slot = task_details.get("incubator_slot")
                future_to_resolve = task_details.get("future")
                
                try:
                    if action == "load":
                        # Ensure microscope is set up for the task if this worker needs to know about it
                        # For now, assuming setup_connections is handled by the main run_cycle or similar trigger.
                        # If transport worker needs to select/ensure microscope, that logic would go here or be passed in task_details.
                        if not self.microscope: # Basic check, might need more sophisticated microscope management
                             raise MicroscopeNotAvailableError("Microscope not available for load operation")
                        await self._execute_load_operation(incubator_slot)
                        if future_to_resolve: future_to_resolve.set_result(True)
                    elif action == "unload":
                        if not self.microscope: # Basic check
                             raise MicroscopeNotAvailableError("Microscope not available for unload operation")
                        await self._execute_unload_operation(incubator_slot)
                        if future_to_resolve: future_to_resolve.set_result(True)
                    else:
                        error_msg = f"Unknown transport action: {action}"
                        logger.warning(f"Transport worker received unknown action: {action}")
                        if future_to_resolve: future_to_resolve.set_exception(ValueError(error_msg))

                except Exception as e:
                    logger.error(f"Transport worker failed for {action} on slot {incubator_slot}: {e}")
                    if future_to_resolve and not future_to_resolve.done():
                        future_to_resolve.set_exception(e)
                
                self.transport_queue.task_done()
                logger.info(f"Transport task {action} for slot {incubator_slot} completed")

            except asyncio.CancelledError:
                logger.info("Transport worker loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Exception in transport worker loop: {e}", exc_info=True)
                # Continue to prevent worker death

    async def _start_transport_worker(self):
        if self._transport_worker_task is None or self._transport_worker_task.done():
            self._transport_worker_task = asyncio.create_task(self._transport_worker_loop())
            logger.info("Transport worker task created and started.")
        else:
            logger.info("Transport worker task already running.")

    async def _stop_transport_worker(self):
        if self._transport_worker_task and not self._transport_worker_task.done():
            logger.info("Stopping transport worker task...")
            self.transport_queue.put_nowait(None) # Send a sentinel to potentially break the loop cleanly if needed
            self._transport_worker_task.cancel()
            try:
                await self._transport_worker_task
            except asyncio.CancelledError:
                logger.info("Transport worker task successfully cancelled.")
            self._transport_worker_task = None
        else:
            logger.info("Transport worker task not running or already stopped.")

    async def _register_self_as_hypha_service(self):
        logger.info(f"Registering orchestrator as a Hypha service with ID '{self.orchestrator_hypha_service_id}' on server '{self.orchestrator_hypha_server_url}' in workspace '{self.workspace}'")
        if not self.token_for_orchestrator_registration:
            logger.error("REEF_WORKSPACE_TOKEN is not set in environment. Cannot register orchestrator service.")
            return

        server_config_for_registration = {
            "server_url": self.orchestrator_hypha_server_url,
            "ping_interval": None,
            "workspace": self.workspace,
            "token": self.token_for_orchestrator_registration
        }
        
        self.orchestrator_hypha_server_connection = await connect_to_server(server_config_for_registration)
        logger.info(f"Successfully connected to Hypha server: {self.orchestrator_hypha_server_url} for orchestrator registration")

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
            "load_plate_from_incubator_to_microscope": self.load_plate_from_incubator_to_microscope, # Will be changed to queue task
            "unload_plate_from_microscope": self.unload_plate_from_microscope, # Will be changed to queue task
        }
        
        registered_service = await self.orchestrator_hypha_server_connection.register_service(service_api)
        logger.info(f"Orchestrator management service registered successfully. Service ID: {registered_service.id}")
        
        # Start the transport worker after successful registration
        await self._start_transport_worker()

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
        
        if not isinstance(new_settings["pending_time_points"], list):
            msg = f"'pending_time_points' must be a list for task '{task_name}'."
            logger.error(msg)
            return {"success": False, "message": msg}

        parsed_pending_time_points_str = []
        if not new_settings["pending_time_points"]:
            logger.warning(f"Task '{task_name}' has an empty 'pending_time_points' list.")
        
        for tp_str in new_settings["pending_time_points"]:
            try:
                datetime.fromisoformat(tp_str) # Validate naive ISO format
                if 'Z' in tp_str or '+' in tp_str.split('T')[-1]: # Basic check for unwanted timezone indicators
                    raise ValueError("Time point string should be naive local time.")
                parsed_pending_time_points_str.append(tp_str)
            except ValueError as ve:
                msg = f"Invalid naive ISO format for a time point in 'pending_time_points' for task '{task_name}': {tp_str} ({ve})"
                logger.error(msg)
                return {"success": False, "message": msg}
        
        # Ensure "imaged_time_points" exists and is a list, defaulting to empty if not provided
        if "imaged_time_points" not in new_settings:
            new_settings["imaged_time_points"] = []
        elif not isinstance(new_settings["imaged_time_points"], list):
            msg = f"'imaged_time_points' must be a list if provided for task '{task_name}'."
            logger.error(msg)
            return {"success": False, "message": msg}
        
        for tp_str in new_settings["imaged_time_points"]: # Also validate imaged if provided
            try:
                datetime.fromisoformat(tp_str)
                if 'Z' in tp_str or '+' in tp_str.split('T')[-1]:
                     raise ValueError("Time point string should be naive local time.")
            except ValueError as ve:
                msg = f"Invalid naive ISO format for a time point in 'imaged_time_points' for task '{task_name}': {tp_str} ({ve})"
                logger.error(msg)
                return {"success": False, "message": msg}

        # Determine status and flags based on the new/updated time points
        has_pending = bool(parsed_pending_time_points_str)
        has_imaged = bool(new_settings.get("imaged_time_points", []))

        current_status = "pending"
        if not has_pending:
            current_status = "completed"
        
        new_settings["imaging_completed"] = not has_pending
        new_settings["imaging_started"] = has_imaged or (not has_pending and has_imaged)

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
                
                op_state = {
                    "status": current_status,
                    "last_updated_by_orchestrator": datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                }

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
        new_task_configs = {}
        raw_config_data = None 

        async with self._config_lock:
            try:
                with open(CONFIG_FILE_PATH, 'r') as f:
                    raw_config_data = json.load(f) 
            except FileNotFoundError:
                logger.error(f"Configuration file {CONFIG_FILE_PATH} not found for simulation.")
                raw_config_data = {"samples": []} 
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {CONFIG_FILE_PATH} for simulation. Will not update tasks from file this cycle.")
                return 

        current_time_naive = datetime.now() # Used for intelligent flag setting, not scheduling here

        for sample_config_from_file in raw_config_data.get("samples", []):
            task_name = sample_config_from_file.get("name")
            settings = sample_config_from_file.get("settings")

            if not task_name or not settings:
                logger.warning(f"Found a sample configuration without a name or settings in {CONFIG_FILE_PATH}. Skipping: {sample_config_from_file}")
                continue

            try:
                pending_datetimes = []
                for tp_str in settings.get("pending_time_points", []):
                    dt_obj = datetime.fromisoformat(tp_str) # Expects naive ISO string
                    pending_datetimes.append(dt_obj)
                pending_datetimes.sort() 

                imaged_datetimes = []
                for tp_str in settings.get("imaged_time_points", []):
                    dt_obj = datetime.fromisoformat(tp_str) # Expects naive ISO string
                    imaged_datetimes.append(dt_obj)
                imaged_datetimes.sort()
                
                # Determine flags based on actual datetime lists
                has_pending = bool(pending_datetimes)
                has_imaged = bool(imaged_datetimes)

                # These flags in settings are for what's WRITTEN to config, orchestrator uses internal datetime lists primarily.
                settings["imaging_completed"] = not has_pending
                settings["imaging_started"] = has_imaged or (not has_pending and has_imaged) # Started if imaged, or completed & imaged

                parsed_settings_config = {
                    "name": task_name,
                    "incubator_slot": settings["incubator_slot"],
                    "allocated_microscope": settings.get("allocated_microscope", SIM_DEFAULT_MICROSCOPE_ID),
                    "imaging_zone": settings["imaging_zone"],
                    "Nx": settings["Nx"],
                    "Ny": settings["Ny"],
                    "illuminate_channels": settings["illuminate_channels"],
                    "do_reflection_af": settings["do_reflection_af"],
                    "pending_datetimes": pending_datetimes, 
                    "imaged_datetimes": imaged_datetimes,
                    # The flags below are derived for internal logic if needed, but primary truth is pending/imaged_datetimes counts
                    "imaging_started_flag": settings["imaging_started"], 
                    "imaging_completed_flag": settings["imaging_completed"]
                }
                new_task_configs[task_name] = parsed_settings_config
            except KeyError as e:
                logger.error(f"Missing key {e} in simulation configuration settings for sample {task_name}. Skipping.")
                continue
            except ValueError as e: # Catch errors from datetime.fromisoformat if string is not naive or malformed
                logger.error(f"Error parsing time strings (ensure they are naive local time) for sample {task_name}: {e}. Skipping.")
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
            
            persisted_status = operational_state_from_file.get("status", "pending")

            # Determine actual status based on current pending_datetimes
            current_actual_status = persisted_status
            if not current_settings_config["pending_datetimes"]:
                current_actual_status = "completed"
            elif persisted_status == "completed" and current_settings_config["pending_datetimes"]:
                 # If file said completed, but now there are pending points (e.g. user added them)
                 current_actual_status = "pending" # Reset to pending
                 logger.info(f"Task '{task_name}' was completed but now has pending points. Resetting status to pending.")
                 a_task_state_changed_for_write = True

            if task_name not in self.tasks:
                logger.info(f"New sim task added: {task_name}")
                self.tasks[task_name] = {
                    "config": current_settings_config, 
                    "status": current_actual_status,
                    "_raw_settings_from_input": copy.deepcopy(sample_config_from_file.get("settings", {}))
                }
                a_task_state_changed_for_write = True # Status might have been determined above

            else: # Task already exists, update it
                existing_task_data = self.tasks[task_name]
                # Check for config changes that might warrant a state reset
                config_changed_significantly = (
                    existing_task_data["config"]["pending_datetimes"] != current_settings_config["pending_datetimes"] or
                    existing_task_data["config"]["imaged_datetimes"] != current_settings_config["imaged_datetimes"] or
                    any(existing_task_data["config"].get(k) != current_settings_config.get(k) 
                        for k in ["incubator_slot", "allocated_microscope", "imaging_zone", "Nx", "Ny"])
                )

                existing_task_data["config"] = current_settings_config # Always update config
                existing_task_data["_raw_settings_from_input"] = copy.deepcopy(sample_config_from_file.get("settings", {}))
                a_task_state_changed_for_write = True # Assume config change implies write needed

                if existing_task_data["status"] != current_actual_status:
                    logger.info(f"Task '{task_name}' status changing from '{existing_task_data['status']}' to '{current_actual_status}' due to config load/re-evaluation.")
                    existing_task_data["status"] = current_actual_status

                if config_changed_significantly and existing_task_data["status"] not in ["pending", "completed"]:
                    # If config changed and task was in an error state, 
                    # and status hasn't already been reset to pending/completed by logic above,
                    # it might need re-evaluation. If new pending points exist, status should become pending.
                    if current_settings_config["pending_datetimes"]:
                        if existing_task_data["status"] != "pending":
                            logger.info(f"Task '{task_name}' had significant config changes while in status '{existing_task_data['status']}'. Resetting to pending as new points exist.")
                            existing_task_data["status"] = "pending"
                    elif not existing_task_data["config"]["imaged_datetimes"]:
                        # No pending, no imaged, but config changed? Should be completed. Or if no pending but imaged.
                         if existing_task_data["status"] != "completed":
                            logger.info(f"Task '{task_name}' had significant config changes. No pending points. Marking completed.")
                            existing_task_data["status"] = "completed"
            
            # Final status check: if a task somehow ends up with status != completed but no pending_datetimes, fix it.
            task_state_dict = self.tasks[task_name]
            if not task_state_dict["config"]["pending_datetimes"] and task_state_dict["status"] != "completed":
                logger.warning(f"Task '{task_name}' has status '{task_state_dict['status']}' but no pending time points. Forcing to 'completed'.")
                task_state_dict["status"] = "completed"
                a_task_state_changed_for_write = True

        if a_task_state_changed_for_write or tasks_to_remove:
            await self._write_tasks_to_config()

    async def _write_tasks_to_config(self):
        """Writes the current state of all tasks back to the configuration file."""
        
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

                settings_to_write["pending_time_points"] = sorted([
                    dt.strftime('%Y-%m-%dT%H:%M:%S') for dt in current_internal_config.get("pending_datetimes", [])
                ])
                settings_to_write["imaged_time_points"] = sorted([
                    dt.strftime('%Y-%m-%dT%H:%M:%S') for dt in current_internal_config.get("imaged_datetimes", [])
                ])

                has_pending = bool(current_internal_config.get("pending_datetimes"))
                has_imaged = bool(current_internal_config.get("imaged_datetimes"))
                
                # Update these flags based on the current truth (pending/imaged datetimes)
                settings_to_write["imaging_completed"] = not has_pending
                settings_to_write["imaging_started"] = has_imaged or (not has_pending and has_imaged)
                
                sample_entry = {
                    "name": task_name,
                    "settings": settings_to_write, 
                    "operational_state": {
                        "status": task_data_internal["status"],
                        "last_updated_by_orchestrator": datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                    }
                }
                output_config_data["samples"].append(sample_entry)
            
            try:
                with open(CONFIG_FILE_PATH, 'w') as f_write:
                    json.dump(output_config_data, f_write, indent=4)
            except IOError as e:
                logger.error(f"Error writing tasks state to {CONFIG_FILE_PATH}: {e}")
                
    async def _update_task_state_and_write_config(self, task_name, status=None, current_tp_to_move_to_imaged: datetime = None):
        """Helper to update task state (including time points) and write to config."""
        if task_name not in self.tasks:
            logger.warning(f"_update_task_state_and_write_config: Task {task_name} not found.")
            return

        changed = False
        task_state = self.tasks[task_name]
        task_config_internal = task_state["config"]

        if status and task_state["status"] != status:
            logger.info(f"Task '{task_name}' status changing from '{task_state['status']}' to '{status}'")
            task_state["status"] = status
            changed = True
        
        if current_tp_to_move_to_imaged:
            if current_tp_to_move_to_imaged in task_config_internal["pending_datetimes"]:
                task_config_internal["pending_datetimes"].remove(current_tp_to_move_to_imaged)
                task_config_internal["imaged_datetimes"].append(current_tp_to_move_to_imaged)
                task_config_internal["imaged_datetimes"].sort() 
                logger.info(f"Moved time point {current_tp_to_move_to_imaged.isoformat()} to imaged for task '{task_name}'.")
                changed = True
            else:
                logger.warning(f"Time point {current_tp_to_move_to_imaged.isoformat()} not found in pending_datetimes for task '{task_name}'. Cannot move.")

        # Update status based on pending points
        if not task_config_internal["pending_datetimes"]: 
            if task_state["status"] != "completed":
                logger.info(f"Task '{task_name}' has no more pending time points. Marking as completed.")
                task_state["status"] = "completed"
                changed = True
        elif status == "completed" and task_config_internal["pending_datetimes"]:
            logger.warning(f"Task '{task_name}' set to completed, but still has pending points. Reverting to pending.")
            task_state["status"] = "pending"
            changed = True

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
        
        # Verify essential mock services are available
        if not self.incubator or not self.robotic_arm:
            raise ServiceConnectionError("Essential mock services (incubator or robotic arm) not available")

    async def check_service_health(self, service, service_type_str):
        """Check if the service is healthy and reset if needed (simulated)"""
        service_name = service.id if hasattr(service, "id") else f"simulated_{service_type_str}_service"
            
        while True:
            try:
                hello_world_result = await service.hello_world()
                if hello_world_result != "Hello world":
                    error_msg = f"{service_name} (sim) hello_world check failed: {hello_world_result}"
                    logger.error(error_msg)
                    raise ServiceConnectionError(error_msg)
                logger.debug(f"Health check passed for {service_name} (sim)")
            except Exception as e:
                logger.error(f"{service_name} (sim) health check failed: {e}")
                logger.info(f"Attempting to reset simulated {service_type_str} service...")
                try:
                    await self.disconnect_single_service(service_type_str)
                    await self.reconnect_single_service(service_type_str) # Pass the type string
                except Exception as reconnect_error:
                    logger.error(f"Failed to reconnect {service_type_str}: {reconnect_error}")
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
        elif service_type == 'orchestrator' and self.orchestrator:
            logger.info(f"Sim: Clearing orchestrator mock instance.")
            # await self.orchestrator.disconnect()
            self.orchestrator = None
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

    async def load_plate_from_incubator_to_microscope(self, incubator_slot: int):
        logger.info(f"API call: Queuing load_plate_from_incubator_to_microscope for slot {incubator_slot}")
        # In a real scenario with Hypha, this method might need to be non-async if Hypha handles the async call wrapper.
        # For now, keeping it async to create and use asyncio.Future.
        op_future = asyncio.get_event_loop().create_future()
        await self.transport_queue.put({
            "action": "load",
            "incubator_slot": incubator_slot,
            "future": op_future
        })
        # The Hypha caller will get this immediate response.
        # The actual result of the transport will be in op_future, if awaited internally or exposed via another method.
        return {"success": True, "message": f"Load task for slot {incubator_slot} queued."}

    async def _execute_load_operation(self, incubator_slot): 
        if self.sample_on_microscope_flag:
            logger.info("Sim (Execute Load): Sample plate already on microscope")
            return # Success case - no need to load again
            
        logger.info(f"Sim (Execute Load): Loading sample from incubator slot {incubator_slot}...")
        
        try:
            await self.incubator.get_sample_from_slot_to_transfer_station(incubator_slot)
            await self.microscope.home_stage()
            await self.robotic_arm.grab_sample_from_incubator()
            await self.robotic_arm.transport_from_incubator_to_microscope1()
            await self.robotic_arm.put_sample_on_microscope1()
            await self.microscope.return_stage()
            
            logger.info("Sim (Execute Load): Sample loaded onto microscope.")
            self.sample_on_microscope_flag = True
            
        except Exception as e:
            error_msg = f"Failed to load sample from slot {incubator_slot}: {e}"
            logger.error(f"Sim (Execute Load): {error_msg}")
            raise TransportOperationError(error_msg) from e

    async def unload_plate_from_microscope(self, incubator_slot: int):
        logger.info(f"API call: Queuing unload_plate_from_microscope for slot {incubator_slot}")
        op_future = asyncio.get_event_loop().create_future()
        await self.transport_queue.put({
            "action": "unload",
            "incubator_slot": incubator_slot,
            "future": op_future
        })
        return {"success": True, "message": f"Unload task for slot {incubator_slot} queued."}

    async def _execute_unload_operation(self, incubator_slot):
        if not self.sample_on_microscope_flag:
            logger.info("Sim (Execute Unload): Sample plate not on microscope")
            return # Success case - nothing to unload
            
        logger.info(f"Sim (Execute Unload): Unloading sample to incubator slot {incubator_slot}...")

        try:
            await self.microscope.home_stage()
            await self.robotic_arm.grab_sample_from_microscope1()
            await self.robotic_arm.transport_from_microscope1_to_incubator()
            await self.robotic_arm.put_sample_on_incubator()
            await self.incubator.put_sample_from_transfer_station_to_slot(incubator_slot)
            await self.microscope.return_stage()
            
            logger.info("Sim (Execute Unload): Sample unloaded from microscope.")
            self.sample_on_microscope_flag = False
            
        except Exception as e:
            error_msg = f"Failed to unload sample to slot {incubator_slot}: {e}"
            logger.error(f"Sim (Execute Unload): {error_msg}")
            raise TransportOperationError(error_msg) from e

    async def run_cycle(self, task_config):
        task_name = task_config["name"]
        incubator_slot = task_config["incubator_slot"]
        action_id = f"SIM_{task_name.replace(' ', '_')}-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        logger.info(f"Sim: Starting cycle for task: {task_name} with action_id: {action_id}")

        # Verify services are available
        if not self.incubator or not self.microscope or not self.robotic_arm:
            logger.error(f"Sim: Services not available for task {task_name}. Attempting reconnect.")
            try:
                await self.setup_connections(target_microscope_id_from_task=task_config["allocated_microscope"])
            except Exception as e:
                error_msg = f"Failed to re-establish service connections for task {task_name}: {e}"
                logger.error(f"Sim: {error_msg}")
                raise ServiceConnectionError(error_msg) from e
                
            if not self.microscope:
                error_msg = f"Microscope {task_config['allocated_microscope']} still not available after reconnection"
                logger.error(f"Sim: {error_msg}")
                raise MicroscopeNotAvailableError(error_msg)

        self.sample_on_microscope_flag = False 

        try:
            # For internal run_cycle, we call the execute methods directly for now.
            # If run_cycle should also queue, these calls would change to self.load_plate_... and await future.
            await self._execute_load_operation(incubator_slot=incubator_slot)
            
            await self.microscope.scan_well_plate_simulated(
                illuminate_channels=task_config["illuminate_channels"],
                do_reflection_af=task_config["do_reflection_af"],
                scanning_zone=task_config["imaging_zone"],
                Nx=task_config["Nx"],
                Ny=task_config["Ny"],
                action_ID=action_id,
            )

            await self._execute_unload_operation(incubator_slot=incubator_slot)
            
            logger.info(f"Sim: Cycle for task {task_name} (action_id: {action_id}) completed successfully.")
            
        except Exception as e:
            logger.error(f"Sim: Cycle failed for task {task_name}: {e}")
            # Attempt cleanup - unload if possible
            try:
                await self._execute_unload_operation(incubator_slot=incubator_slot)
                logger.info(f"Sim: Cleanup unload completed for task {task_name} after cycle failure.")
            except Exception as cleanup_error:
                logger.error(f"Sim: Cleanup unload also failed for task {task_name}: {cleanup_error}")
            
            # Re-raise the original exception
            raise

    async def run_time_lapse(self):
        logger.info("Simulated Orchestrator run_time_lapse started.")
        last_config_read_time = 0

        while True:
            current_time_naive = datetime.now()

            if (asyncio.get_event_loop().time() - last_config_read_time) > CONFIG_READ_INTERVAL:
                await self._load_and_update_tasks()
                last_config_read_time = asyncio.get_event_loop().time()

            next_task_to_run = None
            earliest_pending_tp_for_selection = None 

            if not self.tasks:
                logger.debug("Sim: No tasks loaded yet.")
            
            eligible_tasks_for_run = []

            for task_name, task_data in list(self.tasks.items()):
                internal_config = task_data["config"]
                status = task_data["status"]
                pending_datetimes = internal_config.get("pending_datetimes", [])

                if status in ["completed", "error"]:
                    continue
                
                if not pending_datetimes:
                    logger.debug(f"Sim: Task '{task_name}' skipped, no pending time points.")
                    if status != "completed":
                        logger.warning(f"Task '{task_name}' has status '{status}' but no pending points. Marking completed.")
                        await self._update_task_state_and_write_config(task_name, status="completed")
                    continue

                earliest_tp_for_this_task = pending_datetimes[0]
                if current_time_naive >= earliest_tp_for_this_task:
                    eligible_tasks_for_run.append((task_name, earliest_tp_for_this_task))
                    logger.debug(f"Sim: Task '{task_name}' is eligible with TP: {earliest_tp_for_this_task.isoformat()}")
                else:
                    logger.debug(f"Sim: Task '{task_name}' not due yet (earliest TP: {earliest_tp_for_this_task.isoformat()}).")
            
            # Select the task with the overall earliest time point from eligible tasks
            if eligible_tasks_for_run:
                eligible_tasks_for_run.sort(key=lambda x: x[1])
                next_task_to_run, earliest_pending_tp_for_selection = eligible_tasks_for_run[0]
                logger.info(f"Sim: Selected task '{next_task_to_run}' for TP: {earliest_pending_tp_for_selection.isoformat()} from eligible tasks.")

            if next_task_to_run and earliest_pending_tp_for_selection:
                self.active_task_name = next_task_to_run
                task_data = self.tasks[self.active_task_name]
                task_config_for_cycle = task_data["config"]
                current_pending_tp_to_process = earliest_pending_tp_for_selection
                
                logger.info(f"Sim: Preparing to run task {self.active_task_name} for time point {current_pending_tp_to_process.isoformat()}. Current state: status='{task_data['status']}'")

                try:
                    await self.setup_connections(target_microscope_id_from_task=task_config_for_cycle["allocated_microscope"])
                except Exception as setup_error:
                    logger.error(f"Sim: Failed to setup connections for task {self.active_task_name}: {setup_error}")
                    await self._update_task_state_and_write_config(
                        self.active_task_name,
                        status="error"
                    )
                    self.active_task_name = None
                    await asyncio.sleep(ORCHESTRATOR_LOOP_SLEEP)
                    continue
                
                if not self.microscope or self.current_microscope_id != task_config_for_cycle["allocated_microscope"]:
                    logger.error(f"Sim: Microscope {task_config_for_cycle['allocated_microscope']} not available for {self.active_task_name}.")
                    await self._update_task_state_and_write_config(
                        self.active_task_name,
                        status="error"
                    )
                    self.active_task_name = None
                    await asyncio.sleep(ORCHESTRATOR_LOOP_SLEEP)
                    continue

                logger.info(f"Sim: Starting cycle for task: {self.active_task_name}, time point: {current_pending_tp_to_process.isoformat()}")
                await self._update_task_state_and_write_config(self.active_task_name, status="active")
                
                try:
                    await self.run_cycle(task_config_for_cycle) 
                    logger.info(f"Sim: Cycle for task {self.active_task_name}, time point {current_pending_tp_to_process.isoformat()} success.")
                    await self._update_task_state_and_write_config(
                        self.active_task_name,
                        status="waiting_for_next_run",
                        current_tp_to_move_to_imaged=current_pending_tp_to_process
                    )
                except Exception as cycle_error:
                    logger.error(f"Sim: Cycle for task {self.active_task_name}, time point {current_pending_tp_to_process.isoformat()} failed: {cycle_error}")
                    await self._update_task_state_and_write_config(
                        self.active_task_name,
                        status="error"
                    )

                self.active_task_name = None
            else:
                if self.active_task_name:
                    logger.warning("Sim: Active task was set but no task selected for run. Clearing active_task_name.")
                    self.active_task_name = None
                
                # Determine minimum wait time before next loop iteration
                min_wait_time = ORCHESTRATOR_LOOP_SLEEP
                next_potential_run_time = None
                for task_data_val in self.tasks.values():
                    if task_data_val["status"] not in ["completed", "error"] and task_data_val["config"]["pending_datetimes"]:
                        earliest_tp = task_data_val["config"]["pending_datetimes"][0]
                        if next_potential_run_time is None or earliest_tp < next_potential_run_time:
                            next_potential_run_time = earliest_tp
                
                if next_potential_run_time and next_potential_run_time > current_time_naive:
                    wait_seconds = (next_potential_run_time - current_time_naive).total_seconds()
                    min_wait_time = max(0.1, min(wait_seconds, ORCHESTRATOR_LOOP_SLEEP))
                    logger.debug(f"Sim: Calculated dynamic sleep: {min_wait_time:.2f}s until next potential task time ({next_potential_run_time.isoformat()})")


                await asyncio.sleep(min_wait_time)

async def main():
    # Removed argparse and --local argument handling
    log_file_name = f"orchestrator-simulation-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    global logger 
    logger = setup_logging(log_file=log_file_name)

    orchestrator = OrchestrationSystem() # Instantiate without local argument
    
    try:
        await orchestrator._register_self_as_hypha_service() # Register orchestrator's own Hypha service
        await orchestrator.run_time_lapse() # Removed round_time
    except KeyboardInterrupt:
        logger.info("Simulated Orchestrator shutting down (KeyboardInterrupt)...")
    finally:
        logger.info("Simulated Orchestrator performing cleanup...")
        if orchestrator:
            await orchestrator._stop_transport_worker() # Stop the worker
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
