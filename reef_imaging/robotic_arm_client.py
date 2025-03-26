import asyncio
from hypha_rpc import connect_to_server
import os

async def call_robot_service(server_url, workspace, token, operation, max_retries=3, timeout=30):
    """
    Call a robotic arm service operation with retry logic and status checking
    """
    for attempt in range(max_retries):
        try:
            # Connect to server
            server = await connect_to_server({
                "server_url": server_url,
                "workspace": workspace,
                "token": token
            })
            
            # Get robotic arm service
            svc = await server.get_service("robotic-arm-control")
            
            # Check operation status
            status = await svc.get_status(operation)
            
            if status == "failed":
                print(f"Operation {operation} failed")
                return False
                
            if status == "not_started":
                print(f"Starting operation: {operation}")
                # Call the operation
                await asyncio.wait_for(getattr(svc, operation)(), timeout=timeout)
                
            # Wait for completion
            while True:
                status = await svc.get_status(operation)
                if status == "finished":
                    print(f"Operation {operation} completed successfully")
                    await svc.reset_status(operation)
                    return True
                elif status == "failed":
                    print(f"Operation {operation} failed")
                    return False
                await asyncio.sleep(1)
                
        except asyncio.TimeoutError:
            print(f"Operation {operation} timed out. Attempt {attempt + 1}/{max_retries}")
        except Exception as e:
            print(f"Error during {operation}: {e}. Attempt {attempt + 1}/{max_retries}")
        
        await asyncio.sleep(5)  # Wait before retry
        
    print(f"Max retries ({max_retries}) reached for {operation}")
    return False

async def main():
    server_url = "https://hypha.aicell.io"
    workspace = "reef-imaging"
    token = os.environ.get("REEF_WORKSPACE_TOKEN")
    
    # Example workflow
    operations = [
        "move_sample_from_microscope1_to_incubator",
        "move_sample_from_incubator_to_microscope1"
    ]
    
    for operation in operations:
        success = await call_robot_service(server_url, workspace, token, operation)
        if not success:
            print(f"Workflow stopped due to failure in operation: {operation}")
            break

if __name__ == "__main__":
    asyncio.run(main()) 