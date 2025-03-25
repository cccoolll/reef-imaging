import asyncio
from hypha_rpc import connect_to_server
import os
import dotenv

dotenv.load_dotenv()

async def call_service_with_retries(server_url, service_id, task_name, max_retries=3, timeout=5):
    retries = 0
    while retries < max_retries:
        try:
            server = await connect_to_server({"server_url": server_url})
            svc = await server.get_service(service_id)

            # Check the status of the task
            status = await svc.get_status(task_name)
            if status == "not_started":
                print(f"Starting the task {task_name}...")
                if task_name == "hello1":
                    await svc.hello1("John")
                elif task_name == "hello2":
                    await svc.hello2("John")
            elif status == "finished":
                print(f"Task {task_name} already finished.")
                return

            # Wait for the task to complete
            await asyncio.sleep(timeout)
            status = await svc.get_status(task_name)
            if status == "finished":
                print(f"Task {task_name} completed successfully.")
                # Reset the status for the next loop
                await svc.reset_status(task_name)
                return

        except Exception as e:
            print(f"Error: {e}. Retrying... ({retries + 1}/{max_retries})")
            retries += 1
            await asyncio.sleep(timeout)

    print(f"Max retries reached for task {task_name}. Terminating.")

async def main_loop(server_url, service_id):
    while True:
        await call_service_with_retries(server_url, service_id, "hello1")
        await call_service_with_retries(server_url, service_id, "hello2")
        # Add a delay between loops if needed
        await asyncio.sleep(1)

if __name__ == "__main__":
    server_url = "https://hypha.aicell.io"
    workspace = "reef-imaging"
    token = os.environ.get("REEF_WORKSPACE_TOKEN")
    asyncio.run(main_loop(server_url, workspace, token))