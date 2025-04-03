import asyncio
import argparse
import os
from hypha_rpc import connect_to_server, login
from dorna2 import Dorna
import dotenv
from pydantic import Field
from hypha_rpc.utils.schema import schema_function

dotenv.load_dotenv()  
ENV_FILE = dotenv.find_dotenv()  
if ENV_FILE:  
    dotenv.load_dotenv(ENV_FILE)  

class RoboticArmService:
    def __init__(self, local):
        self.local = local
        self.server_url = "http://localhost:9527" if local else "https://hypha.aicell.io"
        self.robot = Dorna()
        self.ip = "192.168.2.20"
        self.connected = False
        self.server = None
        self.service_id = "robotic-arm-control"  # Define service ID here
        # Add task status tracking
        self.task_status = {
            "move_sample_from_microscope1_to_incubator": "not_started",
            "move_sample_from_incubator_to_microscope1": "not_started",
            "grab_sample_from_microscope1": "not_started",
            "grab_sample_from_incubator": "not_started",
            "put_sample_on_microscope1": "not_started",
            "put_sample_on_incubator": "not_started",
            "transport_from_incubator_to_microscope1": "not_started",
            "transport_from_microscope1_to_incubator": "not_started",
            "connect": "not_started",
            "disconnect": "not_started",
            "halt": "not_started",
            "set_alarm": "not_started",
            "get_all_joints": "not_started",
            "get_all_positions": "not_started",
            "light_on": "not_started",
            "light_off": "not_started"
        }

    async def check_service_health(self):
        """Check if the service is healthy and rerun setup if needed"""
        while True:
            try:
                # Try to get the service status
                if self.service_id:
                    service = await self.server.get_service(self.service_id)
                    # Try a simple operation to verify service is working
                    await service.hello_world()
                    #print("Service health check passed")
                else:
                    print("Service ID not set, waiting for service registration")
            except Exception as e:
                print(f"Service health check failed: {e}")
                print("Attempting to rerun setup...")
                # Clean up Hypha service-related connections and variables
                try:
                    if self.server:
                        await self.server.disconnect()
                except Exception as disconnect_error:
                    print(f"Error during disconnect: {disconnect_error}")
                finally:
                    self.server = None

                while True:
                    try:
                        # Rerun the setup method to reset Hypha service
                        await self.setup()
                        print("Setup successful")
                        break  # Exit the loop if setup is successful
                    except Exception as setup_error:
                        print(f"Failed to rerun setup: {setup_error}")
                        await asyncio.sleep(30)  # Wait before retrying
            
            await asyncio.sleep(30)  # Check every half minute

    async def start_hypha_service(self, server):
        self.server = server
        svc = await server.register_service({
            "name": "Robotic Arm Control",
            "id": self.service_id,  # Use the defined service ID
            "config": {
                "visibility": "public",
                "run_in_executor": True
            },
            "hello_world": self.hello_world,
            "move_sample_from_microscope1_to_incubator": self.move_sample_from_microscope1_to_incubator,
            "move_sample_from_incubator_to_microscope1": self.move_sample_from_incubator_to_microscope1,
            "grab_sample_from_microscope1": self.grab_sample_from_microscope1,
            "grab_sample_from_incubator": self.grab_sample_from_incubator,
            "put_sample_on_microscope1": self.put_sample_on_microscope1,
            "put_sample_on_incubator": self.put_sample_on_incubator,
            "transport_from_incubator_to_microscope1": self.transport_from_incubator_to_microscope1,
            "transport_from_microscope1_to_incubator": self.transport_from_microscope1_to_incubator,
            "connect": self.connect,
            "disconnect": self.disconnect,
            "halt": self.halt,
            "get_all_joints": self.get_all_joints,
            "get_all_positions": self.get_all_positions,
            # Add status functions
            "get_task_status": self.get_task_status,
            "reset_task_status": self.reset_task_status,
            "reset_all_task_status": self.reset_all_task_status,
            "set_alarm": self.set_alarm,
            "light_on": self.light_on,
            "light_off": self.light_off,
            "get_actions": self.get_actions,
            "execute_action": self.execute_action
        })

        print(f"Robotic arm control service registered at workspace: {server.config.workspace}, id: {svc.id}")
        print(f'You can use this service using the service id: {svc.id}')
        id = svc.id.split(":")[1]
        print(f"You can also test the service via the HTTP proxy: {self.server_url}/{server.config.workspace}/services/{id}/move_sample_from_microscope1_to_incubator")

        # Start the health check task
        asyncio.create_task(self.check_service_health())

    async def setup(self):
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

    def get_task_status(self, task_name):
        """Get the status of a specific task"""
        return self.task_status.get(task_name, "unknown")

    def reset_task_status(self, task_name):
        """Reset the status of a specific task"""
        if task_name in self.task_status:
            self.task_status[task_name] = "not_started"
    
    def reset_all_task_status(self):
        """Reset the status of all tasks"""
        for task_name in self.task_status:
            self.task_status[task_name] = "not_started"

    def hello_world(self):
        """Hello world"""
        task_name = "hello_world"
        self.task_status[task_name] = "started"
        self.task_status[task_name] = "finished"
        return "Hello world"

    @schema_function(skip_self=True)
    def connect(self):
        """
        Connect and occupy the robot, so that it can be controlled.
        Returns: bool
        """
        self.task_status["connect"] = "started"
        try:
            self.robot.connect(self.ip)
            self.connected = True
            self.task_status["connect"] = "finished"
            print("Connected to robot")
            return True
        except Exception as e:
            self.task_status["connect"] = "failed"
            print(f"Failed to connect: {e}")
            return False

    @schema_function(skip_self=True)
    def disconnect(self):
        """
        Disconnect the robot, so that it can be used by other clients.
        Returns: bool
        """
        self.task_status["disconnect"] = "started"
        try:
            self.robot.close()
            self.connected = False
            self.task_status["disconnect"] = "finished"
            print("Disconnected from robot")
            return True
        except Exception as e:
            self.task_status["disconnect"] = "failed"
            print(f"Failed to disconnect: {e}")
            return False

    @schema_function(skip_self=True)
    def set_motor(self, state: int=Field(1, description="Enable or disable the motor, 1 for enable, 0 for disable")):
        if not self.connected:
            self.connect()
        self.robot.set_motor(state)
        return f"Motor set to {state}"

    @schema_function(skip_self=True)
    def play_script(self, script_path):
        if not self.connected:
            self.connect()
        result = self.robot.play_script(script_path)
        if result != 2:
            raise Exception("Error playing script")
        else:
            return "Script played"
    
    @schema_function(skip_self=True)
    def get_all_joints(self):
        """
        Get the current position of all joints
        Returns: dict
        """
        self.task_status["get_all_joints"] = "started"
        try:
            if not self.connected:
                self.connect()
            result = self.robot.get_all_joint()
            self.task_status["get_all_joints"] = "finished"
            return result
        except Exception as e:
            self.task_status["get_all_joints"] = "failed"
            print(f"Failed to get all joints: {e}")
            return {}

    @schema_function(skip_self=True)
    def get_all_positions(self):
        """
        Get the current position of all joints
        Returns: dict
        """
        self.task_status["get_all_positions"] = "started"
        try:
            if not self.connected:
                self.connect()
            result = self.robot.get_all_pose()
            self.task_status["get_all_positions"] = "finished"
            return result
        except Exception as e:
            self.task_status["get_all_positions"] = "failed"
            print(f"Failed to get all positions: {e}")
            return {}

    @schema_function(skip_self=True)
    def move_sample_from_microscope1_to_incubator(self):
        """
        Move sample from microscope1 to incubator, the microscope need to be homed before
        Returns: bool
        """
        task_name = "move_sample_from_microscope1_to_incubator"
        self.task_status[task_name] = "started"
        self.set_motor(1)
        try:
            self.play_script("paths/microscope1_to_incubator.txt")
            print("Sample moved from microscope1 to incubator")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to move sample from microscope1 to incubator: {e}")
            return False

    @schema_function(skip_self=True)
    def move_sample_from_incubator_to_microscope1(self):
        """
        Move sample from incubator to microscope1, microscope need to be homed before
        Returns: bool
        """
        task_name = "move_sample_from_incubator_to_microscope1"
        self.task_status[task_name] = "started"
        self.set_motor(1)
        try:
            self.play_script("paths/incubator_to_microscope1.txt")
            print("Sample moved from incubator to microscope1")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to move sample from incubator to microscope1: {e}")
            return False

    @schema_function(skip_self=True)
    def grab_sample_from_microscope1(self):
        """
        Transport a sample from microscope1 to the incubator
        Returns: bool
        """
        task_name = "grab_sample_from_microscope1"
        self.task_status[task_name] = "started"
        self.set_motor(1)
        try:
            self.play_script("paths/grab_from_microscope1.txt")
            print("Sample grabbed from microscope1")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to grab sample from microscope1: {e}")
            return False

    @schema_function(skip_self=True)
    def grab_sample_from_incubator(self):
        """
        Grab a sample from the incubator
        Returns: bool
        """
        task_name = "grab_sample_from_incubator"
        self.task_status[task_name] = "started"
        self.set_motor(1)
        try:
            self.play_script("paths/grab_from_incubator.txt")
            print("Sample grabbed from incubator")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to grab sample from incubator: {e}")
            return False

    @schema_function(skip_self=True)
    def put_sample_on_microscope1(self):
        """
        Place a sample on microscope1
        Returns: bool
        """
        task_name = "put_sample_on_microscope1"
        self.task_status[task_name] = "started"
        self.set_motor(1)
        try:
            self.play_script("paths/put_on_microscope1.txt")
            print("Sample placed on microscope1")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to put sample on microscope1: {e}")
            return False

    @schema_function(skip_self=True)
    def put_sample_on_incubator(self):
        """
        Place a sample on the incubator.
        Returns: bool
        """
        task_name = "put_sample_on_incubator"
        self.task_status[task_name] = "started"
        self.set_motor(1)
        try:
            self.play_script("paths/put_on_incubator.txt")
            print("Sample placed on incubator")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to put sample on incubator: {e}")
            return False

    @schema_function(skip_self=True)
    def transport_from_incubator_to_microscope1(self):
        """
        Transport a sample from the incubator to microscope1
        Returns: bool
        """
        task_name = "transport_from_incubator_to_microscope1"
        self.task_status[task_name] = "started"
        self.set_motor(1)
        try:
            self.play_script("paths/transport_from_incubator_to_microscope1.txt")
            print("Sample moved from incubator to microscope1")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to transport sample from incubator to microscope1: {e}")
            return False

    @schema_function(skip_self=True)
    def transport_from_microscope1_to_incubator(self):
        """
        Transport a sample from microscope1 to the incubator
        Returns: bool
        """
        task_name = "transport_from_microscope1_to_incubator"
        self.task_status[task_name] = "started"
        self.set_motor(1)
        try:
            self.play_script("paths/transport_from_microscope1_to_incubator.txt")
            print("Sample moved from microscope1 to incubator")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to transport sample from microscope1 to incubator: {e}")
            return False

    @schema_function(skip_self=True)
    def halt(self):
        """
        Halt/stop the robot, stop all the movements
        Returns: bool
        """
        task_name = "halt"
        self.task_status[task_name] = "started"
        try:
            if not self.connected:
                self.connect()
            self.robot.halt()
            print("Robot halted")
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to halt robot: {e}")
            return False
    
    @schema_function(skip_self=True)
    def set_alarm(self, state: int=Field(1, description="Enable or disable the alarm, 1 for enable, 0 for disable")):
        """
        Set the alarm state
        """
        task_name = "set_alarm"
        self.task_status[task_name] = "started"
        try:
            self.robot.set_alarm(state)
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to set alarm: {e}")
            return False

    @schema_function(skip_self=True)
    def light_on(self):
        """
        Turn on the light
        """
        task_name = "light_on"
        self.task_status[task_name] = "started"
        try:
            self.robot.set_output(7, 0)
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to turn on light: {e}")
            return False

    @schema_function(skip_self=True)
    def light_off(self):    
        """
        Turn off the light
        """
        task_name = "light_off"
        self.task_status[task_name] = "started"
        try:
            self.robot.set_output(7, 1)
            self.task_status[task_name] = "finished"
            return True
        except Exception as e:
            self.task_status[task_name] = "failed"
            print(f"Failed to turn off light: {e}")
            return False

    @schema_function(skip_self=True)
    def get_actions(self):
        """
        Get a list of predefined actions that can be executed by the robot.
        Each action has a name, description, and a list of positions.
        Returns: dict
        """
        import os
        import json
        
        actions = []
        paths_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paths")
        
        # Define action mappings
        action_mappings = {
            "grab_from_incubator.txt": {
                "name": "Grab from Incubator",
                "description": "Grab a sample from the incubator",
                "id": "grab_from_incubator"
            },
            "put_on_incubator.txt": {
                "name": "Put on Incubator",
                "description": "Place a sample on the incubator",
                "id": "put_on_incubator"
            },
            "put_on_microscope1.txt": {
                "name": "Put on Microscope 1",
                "description": "Place a sample on microscope 1",
                "id": "put_on_microscope1"
            },
            "grab_from_microscope1.txt": {
                "name": "Grab from Microscope 1",
                "description": "Grab a sample from microscope 1",
                "id": "grab_from_microscope1"
            },
            "transport_from_incubator_to_microscope1.txt": {
                "name": "Transport from Incubator to Microscope 1",
                "description": "Transport a sample from the incubator to microscope 1",
                "id": "transport_from_incubator_to_microscope1"
            },
            "transport_from_microscope1_to_incubator.txt": {
                "name": "Transport from Microscope 1 to Incubator",
                "description": "Transport a sample from microscope 1 to the incubator",
                "id": "transport_from_microscope1_to_incubator"
            },
            "incubator_to_microscope1.txt": {
                "name": "Move from Incubator to Microscope 1",
                "description": "Move a sample from the incubator to microscope 1 (complete sequence)",
                "id": "incubator_to_microscope1"
            },
            "microscope1_to_incubator.txt": {
                "name": "Move from Microscope 1 to Incubator",
                "description": "Move a sample from microscope 1 to the incubator (complete sequence)",
                "id": "microscope1_to_incubator"
            }
        }
        
        # Process each path file
        for filename, action_info in action_mappings.items():
            file_path = os.path.join(paths_dir, filename)
            if os.path.exists(file_path):
                positions = []
                speeds = []
                
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            try:
                                cmd = json.loads(line)
                                if cmd.get('cmd') == 'jmove':
                                    # Extract position data
                                    pos = [
                                        cmd.get('j0', 0),
                                        cmd.get('j1', 0),
                                        cmd.get('j2', 0),
                                        cmd.get('j3', 0),
                                        cmd.get('j4', 0),
                                        cmd.get('j5', 0)
                                    ]
                                    positions.append(pos)
                                    
                                    # Extract speed if available, otherwise use default of 20
                                    speed = cmd.get('vel', 20)
                                    speeds.append(speed)
                            except json.JSONDecodeError:
                                continue
                
                # Create action object
                action = {
                    "name": action_info["name"],
                    "description": action_info["description"],
                    "id": action_info["id"],
                    "positions": positions,
                    "speeds": speeds
                }
                
                actions.append(action)
        
        return {"actions": actions}

    @schema_function(skip_self=True)
    def execute_action(self, action_id):
        """
        Execute a predefined action by its ID
        Returns: bool
        """
        if not self.connected:
            self.connect()
            
        # Map action IDs to script paths
        action_to_script = {
            "grab_from_incubator": "paths/grab_from_incubator.txt",
            "put_on_incubator": "paths/put_on_incubator.txt",
            "put_on_microscope1": "paths/put_on_microscope1.txt",
            "grab_from_microscope1": "paths/grab_from_microscope1.txt",
            "transport_from_incubator_to_microscope1": "paths/transport_from_incubator_to_microscope1.txt",
            "transport_from_microscope1_to_incubator": "paths/transport_from_microscope1_to_incubator.txt",
            "incubator_to_microscope1": "paths/incubator_to_microscope1.txt",
            "microscope1_to_incubator": "paths/microscope1_to_incubator.txt"
        }
        
        if action_id not in action_to_script:
            print(f"Unknown action ID: {action_id}")
            return False
            
        script_path = action_to_script[action_id]
        try:
            self.set_motor(1)
            result = self.robot.play_script(script_path)
            if result != 2:
                raise Exception("Error playing script")
            print(f"Action {action_id} executed successfully")
            return True
        except Exception as e:
            print(f"Failed to execute action {action_id}: {e}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Hypha service for the robotic arm.")
    parser.add_argument('--local', action='store_true', help="Use localhost as server URL")
    args = parser.parse_args()

    robotic_arm_service = RoboticArmService(local=args.local)

    loop = asyncio.get_event_loop()

    async def main():
        try:
            await robotic_arm_service.setup()
        except Exception as e:
            print(f"Error setting up robotic arm service: {e}")
            raise e

    loop.create_task(main())
    loop.run_forever()
