import asyncio
import json
import os
import time
import random
from hypha_rpc import connect_to_server, login
from pydantic import Field
from hypha_rpc.utils.schema import schema_function
import logging
import dotenv
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiment_server.log")  # Log to a file
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()  
ENV_FILE = dotenv.find_dotenv()  
if ENV_FILE:  
    dotenv.load_dotenv(ENV_FILE)
    
class ExperimentService:
    def __init__(self, local=False):
        self.local = local
        self.server_url = "http://localhost:9527" if local else "https://hypha.aicell.io"
        self.server = None
        self.running = False
        
        # State file path for persistence
        self.state_file = os.path.join(os.path.dirname(__file__), "experiment_state.json")
        
        # Load or initialize state
        self.state = self.load_state()
    
    def load_state(self):
        """Load the persisted state or initialize a new one"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading state: {e}")
            logger.error(f"Error loading state: {e}")
        
        # Default state
        return {
            "last_completed_operation": None,
            "current_operation": None,
            "operation_count": 0,
            "status": "idle",
            "timestamp": time.time()
        }
    
    def save_state(self):
        """Persist the current state to disk"""
        try:
            self.state["timestamp"] = time.time()
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"Error saving state: {e}")
            logger.error(f"Error saving state: {e}")
    
    @schema_function(skip_self=True)
    def get_state(self):
        """
        Get the current state of the experiment service
        Returns: A dictionary with the current state
        """
        return self.state
    
    @schema_function(skip_self=True)
    def get_operation_status(self, operation_id: str):
        """
        Get the status of a specific operation
        Returns: A dictionary with the operation status
        """
        if self.state["current_operation"] == operation_id:
            return {"status": "in_progress"}
        elif self.state["last_completed_operation"] == operation_id:
            return {"status": "completed"}
        else:
            return {"status": "not_started"}
    
    @schema_function(skip_self=True)
    def long_operation_1(self, duration: int = Field(3, description="Operation duration in seconds")):
        """
        Simulate a long-running operation (e.g., grab sample)
        Returns: Operation result
        """
        # Check if already running
        if self.state["status"] != "idle":
            return {
                "success": False, 
                "message": f"Service is busy with: {self.state['current_operation']}"
            }
        
        # Update state to indicate operation is starting
        operation_id = f"operation_1_{int(time.time())}"
        self.state["status"] = "busy"
        self.state["current_operation"] = operation_id
        self.state["operation_count"] += 1
        self.save_state()
        
        print(f"Starting long_operation_1 (ID: {operation_id}, duration: {duration}s)")
        logger.info(f"Starting long_operation_1 (ID: {operation_id}, duration: {duration}s)")
        
        try:
            # Simulate long operation
            time.sleep(duration)
            
            # Update state after successful completion
            self.state["status"] = "idle"
            self.state["last_completed_operation"] = operation_id
            self.state["current_operation"] = None
            self.save_state()
            
            print(f"Completed long_operation_1 (ID: {operation_id})")
            logger.info(f"Completed long_operation_1 (ID: {operation_id})")
            return {
                "success": True, 
                "message": f"Operation {operation_id} completed successfully", 
                "operation_id": operation_id
            }
            
        except Exception as e:
            print(f"Error in long_operation_1: {e}")
            logger.error(f"Error in long_operation_1: {e}")
            return {
                "success": False, 
                "message": f"Operation failed: {str(e)}", 
                "operation_id": operation_id
            }
    
    @schema_function(skip_self=True)
    def long_operation_2(self, duration: int = Field(5, description="Operation duration in seconds")):
        """
        Simulate another long-running operation (e.g., transport sample)
        Returns: Operation result
        """
        # Similar implementation to long_operation_1
        # Check if already running
        if self.state["status"] != "idle":
            return {
                "success": False, 
                "message": f"Service is busy with: {self.state['current_operation']}"
            }
        
        # Update state to indicate operation is starting
        operation_id = f"operation_2_{int(time.time())}"
        self.state["status"] = "busy"
        self.state["current_operation"] = operation_id
        self.state["operation_count"] += 1
        self.save_state()
        
        print(f"Starting long_operation_2 (ID: {operation_id}, duration: {duration}s)")
        logger.info(f"Starting long_operation_2 (ID: {operation_id}, duration: {duration}s)")
        
        try:
            # Simulate long operation
            time.sleep(duration)
            
            # Update state after successful completion
            self.state["status"] = "idle"
            self.state["last_completed_operation"] = operation_id
            self.state["current_operation"] = None
            self.save_state()
            
            print(f"Completed long_operation_2 (ID: {operation_id})")
            logger.info(f"Completed long_operation_2 (ID: {operation_id})")
            return {
                "success": True, 
                "message": f"Operation {operation_id} completed successfully", 
                "operation_id": operation_id
            }
            
        except Exception as e:
            print(f"Error in long_operation_2: {e}")
            logger.error(f"Error in long_operation_2: {e}")
            return {
                "success": False, 
                "message": f"Operation failed: {str(e)}", 
                "operation_id": operation_id
            }
    
    @schema_function(skip_self=True)
    def reset_state(self):
        """
        Reset the service state (for testing purposes)
        Returns: Success status
        """
        self.state = {
            "last_completed_operation": None,
            "current_operation": None,
            "operation_count": 0,
            "status": "idle",
            "timestamp": time.time()
        }
        self.save_state()
        return {"success": True, "message": "State reset successfully"}

    async def start_hypha_service(self, server):
        """Register the experiment service with Hypha"""
        self.server = server
        self.running = True
        
        svc = await server.register_service({
            "name": "Experiment Service",
            "id": "experiment-service",
            "config": {
                "visibility": "public",
                "run_in_executor": True
            },
            "get_state": self.get_state,
            "long_operation_1": self.long_operation_1,
            "long_operation_2": self.long_operation_2,
            "reset_state": self.reset_state
        })

        print(f"Experiment service registered at workspace: {server.config.workspace}, id: {svc.id}")
        logger.info(f"Experiment service registered at workspace: {server.config.workspace}, id: {svc.id}")
        print(f"You can use this service using the service id: {svc.id}")
        id = svc.id.split(":")[1]
        print(f"You can also test the service via the HTTP proxy: {self.server_url}/{server.config.workspace}/services/{id}")
        return svc

    async def shutdown(self):
        """Gracefully shutdown the service"""
        self.running = False
        if self.server:
            await self.server.disconnect()
        print("Service shutdown complete")
        logger.info("Service shutdown complete")

    async def setup(self):
        """Set up the connection to the Hypha server"""
        if self.local:
            token = None
            server = await connect_to_server({"server_url": self.server_url, "token": token, "ping_interval": None})
        else:
            try:
                token = os.environ.get("REEF_WORKSPACE_TOKEN")
            except:
                token = await login({"server_url": self.server_url})
            server = await connect_to_server({"server_url": self.server_url, "token": token, "workspace": "reef-imaging", "ping_interval": None})

        await self.start_hypha_service(server)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start the experimental Hypha service.")
    parser.add_argument('--local', action='store_true', help="Use localhost as server URL")
    args = parser.parse_args()

    experiment_service = ExperimentService(local=args.local)
    loop = asyncio.get_event_loop()

    async def main():
        try:
            await experiment_service.setup()
            # Keep running until interrupted
            while experiment_service.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Received shutdown signal")
            logger.info("Received shutdown signal")
        except Exception as e:
            print(f"Error in experiment service: {e}")
            logger.error(f"Error in experiment service: {e}")
            print(traceback.format_exc())
            logger.error(traceback.format_exc())
        finally:
            await experiment_service.shutdown()

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("Program terminated by user")
        logger.info("Program terminated by user")
    finally:
        loop.close()