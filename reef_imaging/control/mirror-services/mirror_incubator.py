import os
import logging
import logging.handlers
import time
import argparse
import asyncio
import traceback
import dotenv
from hypha_rpc import login, connect_to_server
from typing import Optional, List

dotenv.load_dotenv()  
ENV_FILE = dotenv.find_dotenv()  
if ENV_FILE:  
    dotenv.load_dotenv(ENV_FILE)  

# Set up logging
def setup_logging(log_file="mirror_incubator_service.log", max_bytes=100000, backup_count=3):
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

class MirrorIncubatorService:
    def __init__(self):
        # Connection to cloud service
        self.cloud_server_url = "https://hypha.aicell.io"
        self.cloud_workspace = "reef-imaging"
        self.cloud_token = os.environ.get("REEF_WORKSPACE_TOKEN")
        self.cloud_service_id = "mirror-incubator-control"
        self.cloud_server = None
        
        # Connection to local service
        self.local_server_url = "http://reef.dyn.scilifelab.se:9527"
        self.local_token = os.environ.get("REEF_LOCAL_TOKEN")
        self.local_service_id = "incubator-control"
        self.local_server = None
        self.local_service = None

        # Setup task tracking
        self.setup_task = None
        
        # Store dynamically created mirror methods
        self.mirrored_methods = {}

    async def connect_to_local_service(self):
        """Connect to the local incubator service"""
        try:
            self.local_server = await connect_to_server({
                "server_url": self.local_server_url, 
                "token": self.local_token,
                "ping_interval": None
            })
            
            # Connect to the local service
            self.local_service = await self.local_server.get_service(self.local_service_id)
            logger.info(f"Connected to local service {self.local_service_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to local service: {e}")
            return False

    def _create_mirror_method(self, method_name, local_method):
        """Create a mirror method that forwards calls to the local service"""
        async def mirror_method(*args, **kwargs):
            try:
                if self.local_service is None:
                    await self.connect_to_local_service()
                
                # Forward the call to the local service
                result = await local_method(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Failed to call {method_name}: {e}")
                raise e
        
        return mirror_method

    def _get_mirrored_methods(self):
        """Dynamically create mirror methods for all callable methods in local_service"""
        if self.local_service is None:
            logger.warning("Cannot create mirror methods: local_service is None")
            return {}
        
        mirrored_methods = {}
        
        # Methods to exclude from mirroring (these are handled specially)
        excluded_methods = {
            'name', 'id', 'config', 'type',  # Service metadata
            '__class__', '__doc__', '__dict__', '__module__',  # Python internals
        }
        
        # Get all attributes from the local service
        for attr_name in dir(self.local_service):
            if attr_name.startswith('_') or attr_name in excluded_methods:
                continue
                
            attr = getattr(self.local_service, attr_name)
            
            # Check if it's callable (a method)
            if callable(attr):
                logger.info(f"Creating incubator mirror method for: {attr_name}")
                mirrored_methods[attr_name] = self._create_mirror_method(attr_name, attr)
        
        logger.info(f"Incubator service: Created {len(mirrored_methods)} mirror methods: {list(mirrored_methods.keys())}")
        return mirrored_methods

    async def check_service_health(self):
        """Check if the service is healthy and rerun setup if needed"""
        logger.info("Starting service health check task")
        while True:
            try:
                # Try to get the service status
                if self.cloud_service_id:
                    service = await self.cloud_server.get_service(self.cloud_service_id)
                    # Try a simple operation to verify service is working
                    hello_world_result = await asyncio.wait_for(service.hello_world(), timeout=10)
                    if hello_world_result != "Hello world":
                        logger.error(f"Service health check failed: {hello_world_result}")
                        raise Exception("Service not healthy")
                else:
                    logger.info("Service ID not set, waiting for service registration")
                    
                # Always check local service regardless of whether it's None
                try:
                    if self.local_service is None:
                        logger.info("Local service connection lost, attempting to reconnect")
                        await self.connect_to_local_service()
                        if self.local_service is None:
                            raise Exception("Failed to connect to local service")
                    
                    #logger.info("Checking local service health...")
                    local_hello_world_result = await asyncio.wait_for(self.local_service.hello_world(), timeout=10)
                    #logger.info(f"Local service response: {local_hello_world_result}")
                    
                    if local_hello_world_result != "Hello world":
                        logger.error(f"Local service health check failed: {local_hello_world_result}")
                        raise Exception("Local service not healthy")
                    
                    #logger.info("Local service health check passed")
                except Exception as e:
                    logger.error(f"Local service health check failed: {e}")
                    self.local_service = None  # Reset connection so it will reconnect next time
                    raise Exception(f"Local service not healthy: {e}")
            except Exception as e:
                logger.error(f"Service health check failed: {e}")
                logger.info("Attempting to rerun setup...")
                # Clean up Hypha service-related connections and variables
                try:
                    if self.cloud_server:
                        await self.cloud_server.disconnect()
                    if self.local_server:
                        await self.local_server.disconnect()
                    if self.setup_task:
                        self.setup_task.cancel()  # Cancel the previous setup task
                except Exception as disconnect_error:
                    logger.error(f"Error during disconnect: {disconnect_error}")
                finally:
                    self.cloud_server = None
                    self.local_server = None
                    self.local_service = None

                retry_count = 0
                while retry_count < 30:
                    try:
                        # Rerun the setup method
                        self.setup_task = asyncio.create_task(self.setup())
                        await self.setup_task
                        logger.info("Setup successful")
                        break  # Exit the loop if setup is successful
                    except Exception as setup_error:
                        logger.error(f"Failed to rerun setup: {setup_error}")
                        await asyncio.sleep(30)  # Wait before retrying
            
            await asyncio.sleep(10)  # Check more frequently (was 30)

    async def start_hypha_service(self, server):
        """Start the Hypha service with dynamically mirrored methods"""
        self.cloud_server = server
        
        # First, get the mirrored methods
        self.mirrored_methods = self._get_mirrored_methods()
        
        # Base service configuration with core methods
        service_config = {
            "name": "Mirror Incubator Control",
            "id": self.cloud_service_id,
            "config": {
                "visibility": "public",
                "run_in_executor": True
            },
            "hello_world": self.hello_world,
        }
        
        # Add all mirrored methods to the service configuration
        service_config.update(self.mirrored_methods)
        
        # Register the service
        svc = await server.register_service(service_config)

        logger.info(
            f"Mirror incubator service (service_id={self.cloud_service_id}) started successfully with {len(self.mirrored_methods)} mirrored methods, available at {self.cloud_server_url}/services"
        )

        logger.info(f'You can use this service using the service id: {svc.id}')
        id = svc.id.split(":")[1]

        logger.info(f"You can also test the service via the HTTP proxy: {self.cloud_server_url}/{server.config.workspace}/services/{id}")

    async def setup(self):
        # Connect to cloud workspace
        logger.info(f"Connecting to cloud workspace {self.cloud_workspace} at {self.cloud_server_url}")
        server = await connect_to_server({
            "server_url": self.cloud_server_url, 
            "token": self.cloud_token, 
            "workspace": self.cloud_workspace,
            "ping_interval": None
        })
        
        # Connect to local service first (needed to get available methods)
        await self.connect_to_local_service()
        
        # Start the cloud service with mirrored methods
        await self.start_hypha_service(server)

    def hello_world(self):
        """Hello world - core service method"""
        return "Hello world"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mirror service for Incubator control."
    )
    args = parser.parse_args()

    mirror_service = MirrorIncubatorService()

    loop = asyncio.get_event_loop()

    async def main():
        try:
            mirror_service.setup_task = asyncio.create_task(mirror_service.setup())
            await mirror_service.setup_task
            # Start the health check task
            asyncio.create_task(mirror_service.check_service_health())
        except Exception:
            traceback.print_exc()

    loop.create_task(main())
    loop.run_forever() 