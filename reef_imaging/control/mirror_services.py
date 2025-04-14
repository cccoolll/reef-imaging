import asyncio
import argparse
import os
import logging
import logging.handlers
import time
import dotenv
from hypha_rpc import connect_to_server, login
import re

# Load environment variables
dotenv.load_dotenv()
ENV_FILE = dotenv.find_dotenv()
if ENV_FILE:
    dotenv.load_dotenv(ENV_FILE)

# Set up logging
def setup_logging(log_file="mirror_services.log", max_bytes=100000, backup_count=3):
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

class MirrorService:
    def __init__(self):
        # Local server settings
        self.local_server_url = "http://localhost:9527"
        self.local_token = os.environ.get("REEF_LOCAL_TOKEN")
        if not self.local_token:
            raise ValueError("REEF_LOCAL_TOKEN environment variable is not set")
        
        # Remote server settings
        self.remote_server_url = "https://hypha.aicell.io"
        self.remote_token = os.environ.get("REEF_WORKSPACE_TOKEN")
        if not self.remote_token:
            raise ValueError("REEF_WORKSPACE_TOKEN environment variable is not set")
        
        self.is_connected = False
        
        # Service connections
        self.local_server = None
        self.remote_server = None
        
        # For tracking registered mirror services on remote server
        self.mirror_services = {}
        
        # For health check tasks
        self.health_check_tasks = []
        
        # Track setup task
        self.setup_task = None
        
        # List of services we've already mirrored to avoid duplication
        self.mirrored_service_ids = set()

    def clean_service_id(self, service_id):
        """Remove workspace and random ID prefixes from service ID"""
        # Match pattern like 'workspace/randomID:actual-service-name'
        match = re.search(r'(?:[^:]+:)?([\w-]+(?:-simulation)?)', service_id)
        if match:
            return match.group(1)
        return service_id  # Return original if no match

    async def connect_to_local_server(self):
        """Connect to the local server"""
        try:
            logger.info(f"Connecting to local server at {self.local_server_url}")
            self.local_server = await connect_to_server({
                "server_url": self.local_server_url,
                "token": self.local_token,
                "ping_interval": None
            })
            logger.info("Connected to local server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to local server: {e}")
            return False

    async def connect_to_remote_server(self):
        """Connect to the remote server"""
        try:
            logger.info(f"Connecting to remote server at {self.remote_server_url}")
            self.remote_server = await connect_to_server({
                "server_url": self.remote_server_url,
                "token": self.remote_token,
                "workspace": "reef-imaging",
                "ping_interval": None
            })
            logger.info("Connected to remote server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to remote server: {e}")
            return False

    async def list_local_services(self):
        """List all services available on the local server"""
        try:
            services = await self.local_server.list_services()
            logger.info(f"Found {len(services)} services on local server")
            return services
        except Exception as e:
            logger.error(f"Failed to list local services: {e}")
            return []
            
    def should_mirror_service(self, service_id, service_type):
        """Determine if a service should be mirrored"""
        # Skip built-in services
        if service_type == "built-in":
            logger.info(f"Skipping built-in service: {service_id}")
            return False
            
        # Skip services with "mirror" in their ID to avoid recursive mirroring
        if "mirror" in service_id:
            logger.info(f"Skipping mirror service: {service_id}")
            return False
            
        # Skip services we've already mirrored
        if service_id in self.mirrored_service_ids:
            logger.info(f"Skipping already mirrored service: {service_id}")
            return False
            
        # Only mirror specific service types
        target_services = ["microscope-control", "robotic-arm-control", "incubator-control"]
        clean_id = self.clean_service_id(service_id)
        
        for target in target_services:
            if target in clean_id:
                return True
                
        logger.info(f"Skipping service that is not in our target list: {service_id}")
        return False

    def mirror_method(self, local_service, method_name):
        """Create a mirror method that forwards calls to the local service"""
        async def mirrored_method(*args, **kwargs):
            try:
                logger.info(f"Forwarding call to {method_name} with args={args}, kwargs={kwargs}")
                result = await getattr(local_service, method_name)(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Error mirroring method {method_name}: {e}")
                raise
        return mirrored_method

    async def register_mirror_service(self, service_id, service_name, service_type, methods):
        """Register a mirror service on the remote server"""
        # Check if we should mirror this service
        if not self.should_mirror_service(service_id, service_type):
            return False
            
        try:
            # Get the local service
            local_service = await self.local_server.get_service(service_id)
            if not local_service:
                logger.error(f"Local service {service_id} not found")
                return False
            
            # Clean service ID to remove workspace and random IDs
            clean_id = self.clean_service_id(service_id)
            mirror_id = f"{clean_id}-mirror"
            
            logger.info(f"Registering mirror service for {service_id}")
            logger.info(f"Using cleaned service ID: {clean_id}")
            logger.info(f"Mirror ID: {mirror_id}")
            
            # Create service configuration
            service_config = {
                "id": mirror_id,
                "name": f"{service_name} Mirror",
                "config": {
                    "visibility": "public",
                    "run_in_executor": True
                }
            }
            
            # Add all methods from the local service
            for method_name in methods:
                service_config[method_name] = self.mirror_method(local_service, method_name)
            
            # Register the service
            mirror_service = await self.remote_server.register_service(service_config)
            self.mirror_services[service_id] = mirror_service
            
            # Add to the set of mirrored services
            self.mirrored_service_ids.add(service_id)
            
            # Log service information
            svc_id = mirror_service.id
            logger.info(f"Mirror service registered with ID: {svc_id}")
            id_part = svc_id.split(":")[1]
            logger.info(f"Access the service at: {self.remote_server_url}/reef-imaging/services/{id_part}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to register mirror service for {service_id}: {e}")
            return False

    async def check_service_health(self, service_id):
        """Check if the mirrored service is healthy and reconnect if needed"""
        while True:
            try:
                if service_id in self.mirror_services and self.mirror_services[service_id]:
                    # Check if the mirror service is still accessible
                    # Try to call hello_world if it exists, otherwise just check if the service exists
                    mirror_service = self.mirror_services[service_id]
                    if hasattr(mirror_service, "hello_world"):
                        await mirror_service.hello_world()
                    else:
                        # Just check if the service exists by accessing some attribute
                        _ = mirror_service.id
                    logger.debug(f"Mirror service for {service_id} is healthy")
            except Exception as e:
                logger.error(f"Health check failed for {service_id} mirror: {e}")
                logger.info(f"Attempting to re-register {service_id} mirror service...")
                
                try:
                    # Get the service information again
                    services = await self.list_local_services()
                    for service in services:
                        if service["id"] == service_id:
                            methods = service.get("methods", [])
                            service_type = service.get("type", "unknown")
                            service_name = service.get("name", service_id)
                            await self.register_mirror_service(service_id, service_name, service_type, methods)
                            break
                except Exception as reg_error:
                    logger.error(f"Failed to re-register {service_id} mirror service: {reg_error}")
            
            await asyncio.sleep(30)  # Check every 30 seconds

    async def setup(self):
        """Connect to servers and set up the mirror services"""
        # Connect to local and remote servers
        local_connected = await self.connect_to_local_server()
        remote_connected = await self.connect_to_remote_server()
        
        if not (local_connected and remote_connected):
            logger.error("Failed to connect to either local or remote server")
            return False
        
        self.is_connected = True
        
        # Get all services from local server
        services = await self.list_local_services()
        
        # Mirror each service
        for service in services:
            service_id = service["id"]
            service_name = service.get("name", service_id)
            service_type = service.get("type", "unknown")
            methods = service.get("methods", [])
            
            # Register the mirror service if it should be mirrored
            success = await self.register_mirror_service(service_id, service_name, service_type, methods)
            if success:
                # Start health check for this service
                task = asyncio.create_task(self.check_service_health(service_id))
                self.health_check_tasks.append(task)
        
        logger.info(f"Successfully mirrored {len(self.mirror_services)} services")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mirror local Hypha services to hypha.aicell.io")
    args = parser.parse_args()

    mirror_service = MirrorService()
    
    loop = asyncio.get_event_loop()
    
    async def main():
        try:
            mirror_service.setup_task = asyncio.create_task(mirror_service.setup())
            await mirror_service.setup_task
            logger.info("Mirror services setup complete")
        except Exception as e:
            logger.error(f"Error setting up mirror services: {e}")
            raise e
    
    loop.create_task(main())
    loop.run_forever() 