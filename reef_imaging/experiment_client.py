import asyncio
import logging
import os
import time
import traceback
from hypha_rpc import connect_to_server, login
import dotenv
import json

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Ensure logs are output to the console
        logging.FileHandler("experiment_client.log")  # Log to a file
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 50000000
RETRY_DELAY = 5  # seconds
SERVER_URL = "https://hypha.aicell.io"  # Change to your actual server URL if needed
WORKSPACE = "reef-imaging"
TOKEN = os.environ.get("REEF_WORKSPACE_TOKEN")

class ExperimentClient:
    """Client for interacting with the experiment service with reconnection handling"""
    
    def __init__(self):
        self.server_url = SERVER_URL
        self.workspace = WORKSPACE
        self.token = TOKEN
        self.server = None
        self.service = None
        self.connected = False
        self.last_operation_id = None
    
    async def connect(self):
        """Connect to the Hypha server and get the experiment service"""
        try:
            print("Attempting to connect to the server...")
            self.server = await connect_to_server({
                "server_url": self.server_url,
                "token": self.token,
                "workspace": self.workspace,
                "ping_interval": None
            })
            
            self.service = await self.server.get_service("experiment-service")
            self.connected = True
            print("Connected to experiment service")
            logger.info("Connected to experiment service")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            logger.error(f"Failed to connect: {e}")
            self.connected = False
            return False
    
    async def ensure_connected(self):
        """Ensure connection is established, with retry logic"""
        if not self.connected:
            retry_count = 0
            while retry_count < MAX_RETRIES:
                print(f"Attempting to connect (attempt {retry_count + 1}/{MAX_RETRIES})...")
                logger.info(f"Attempting to connect (attempt {retry_count + 1}/{MAX_RETRIES})...")
                if await self.connect():
                    return True
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
            raise Exception(f"Failed to connect after {MAX_RETRIES} attempts")
        return True
    
    async def execute_operation_with_retry(self, operation_name, *args, **kwargs):
        """Execute a service operation with reconnection and retry logic"""
        await self.ensure_connected()
        
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                # Store which operation we're about to execute
                self.last_operation_id = f"{operation_name}_{int(time.time())}"
                print(f"Executing {operation_name} (attempt {retry_count + 1}/{MAX_RETRIES})...")
                
                # Get the method from the service
                method = getattr(self.service, operation_name)
                
                # Execute the method
                result = await method(*args, **kwargs)
                print(f"Operation result: {result}")
                logger.info(f"Operation result: {result}")
                return result
                
            except Exception as e:
                print(f"Error during {operation_name}: {e}")
                logger.error(f"Error during {operation_name}: {e}")
                traceback.print_exc()
                
                # Check if it's a connection error
                if "WebSocket connection" in str(e) or "server rejected" in str(e):
                    print("Connection issue detected, checking operation status...")
                    logger.warning("Connection issue detected, checking operation status...")
                    
                    # Try to reconnect
                    self.connected = False
                    await asyncio.sleep(RETRY_DELAY)
                    if await self.ensure_connected():
                        # Check if the operation completed despite the connection error
                        try:
                            state = await self.service.get_state()
                            print(f"Service state after reconnection: {state}")
                            logger.info(f"Service state after reconnection: {state}")
                            
                            # Check if our operation might have completed
                            if state["last_completed_operation"] and operation_name in state["last_completed_operation"]:
                                print(f"Operation {operation_name} may have completed despite connection error")
                                logger.info(f"Operation {operation_name} may have completed despite connection error")
                                return {
                                    "success": True,
                                    "message": f"Operation likely completed (verified after reconnection)",
                                    "operation_id": state["last_completed_operation"],
                                    "reconnected": True
                                }
                            
                            # Check if operation is still in progress
                            if state["status"] == "busy" and state["current_operation"] and operation_name in state["current_operation"]:
                                print(f"Operation {operation_name} is still in progress")
                                logger.info(f"Operation {operation_name} is still in progress")
                                # Wait for it to complete or retry
                                await asyncio.sleep(RETRY_DELAY * 2)
                                continue
                            
                            # If operation didn't complete and isn't running, we should retry
                            print(f"Operation {operation_name} needs to be retried")
                            logger.info(f"Operation {operation_name} needs to be retried")
                        except Exception as check_error:
                            print(f"Error checking operation status: {check_error}")
                            logger.error(f"Error checking operation status: {check_error}")
                
                # Increment retry counter and try again
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    print(f"Maximum retry attempts reached for {operation_name}")
                    logger.error(f"Maximum retry attempts reached for {operation_name}")
                    raise Exception(f"Failed to execute {operation_name} after {MAX_RETRIES} attempts: {str(e)}")
                
                await asyncio.sleep(RETRY_DELAY)
    
    async def run_test(self):
        """Run a test sequence of operations"""
        print("Running test sequence")
        logger.info("Running test sequence")
        try:
            # Reset state before starting
            await self.execute_operation_with_retry("reset_state")
            
            # Run first operation (shorter duration)
            result1 = await self.execute_operation_with_retry("long_operation_1", duration=8)
            print(f"Operation 1 result: {result1}")
            
            # Run second operation (longer duration with higher crash potential)
            result2 = await self.execute_operation_with_retry("long_operation_2", duration=12)
            print(f"Operation 2 result: {result2}")
            
            # Get final state
            final_state = await self.execute_operation_with_retry("get_state")
            print(f"Final state: {final_state}")
            
            return {
                "success": True,
                "results": {
                    "operation1": result1,
                    "operation2": result2,
                    "final_state": final_state
                }
            }
            
        except Exception as e:
            print(f"Test sequence failed: {e}")
            logger.error(f"Test sequence failed: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

async def main():
    """Main function to run the client"""
    client = ExperimentClient()
    
    # Run test continuously until interrupted
    while True:
        try:
            print("Starting experiment test sequence")
            logger.info("Starting experiment test sequence")
            result = await client.run_test()
            
            if result["success"]:
                print("Test sequence completed successfully")
                logger.info("Test sequence completed successfully")
                print(f"Results: {result['results']}")
            else:
                print(f"Test sequence failed: {result['error']}")
                logger.error(f"Test sequence failed: {result['error']}")
            
            # Wait before starting next test sequence
            print("Waiting 5 seconds before next test sequence...")
            logger.info("Waiting 5 seconds before next test sequence...")
            await asyncio.sleep(5)
            
        except KeyboardInterrupt:
            print("Received keyboard interrupt, shutting down...")
            logger.info("Received keyboard interrupt, shutting down...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            logger.error(f"Unexpected error: {e}")
            print(traceback.format_exc())
            logger.error(traceback.format_exc())
            # Wait before retry on error
            await asyncio.sleep(10)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program terminated by user")
        logger.info("Program terminated by user")