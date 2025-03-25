import asyncio
from hypha_rpc import connect_to_server
import os
import dotenv
import logging
import sys

dotenv.load_dotenv()

# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "experiment_client.log")
txt_file = os.path.join(log_dir, "experiment_client.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def write_to_txt_file(message):
    with open(txt_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

async def call_service_with_retries(server_url, service_id, task_name, workspace, token, max_retries=100, timeout=5):
    retries = 0
    while retries < max_retries:
        try:
            server = await connect_to_server({"server_url": server_url, "workspace": workspace, "token": token})
            svc = await server.get_service(service_id)

            # Check the status of the task
            status = await svc.get_status(task_name)
            if status == "failed":
                message = f"Task {task_name} failed. Stopping execution."
                print(message)
                logger.error(message)
                write_to_txt_file(message)
                sys.exit(1)  # Exit the program with a non-zero status

            if status == "not_started":
                message = f"Starting the task {task_name}..."
                print(message)
                logger.info(message)
                write_to_txt_file(message)
                if task_name == "hello1":
                    await asyncio.wait_for(svc.hello1("John"), timeout=timeout)
                elif task_name == "hello2":
                    await asyncio.wait_for(svc.hello2("John"), timeout=timeout)
            elif status == "finished":
                message = f"Task {task_name} already finished."
                print(message)
                logger.info(message)
                write_to_txt_file(message)
                return

            # Wait for the task to complete
            status = await asyncio.wait_for(svc.get_status(task_name), timeout=timeout)
            if status == "finished":
                message = f"Task {task_name} completed successfully."
                print(message)
                logger.info(message)
                write_to_txt_file(message)
                # Reset the status for the next loop
                await svc.reset_status(task_name)
                return

        except asyncio.TimeoutError:
            message = f"Operation {task_name} timed out. Retrying... ({retries + 1}/{max_retries})"
            print(message)
            logger.warning(message)
            write_to_txt_file(message)
        except Exception as e:
            message = f"Error: {e}. Retrying... ({retries + 1}/{max_retries})"
            print(message)
            logger.error(message)
            write_to_txt_file(message)
        retries += 1
        await asyncio.sleep(timeout)

    message = f"Max retries reached for task {task_name}. Terminating."
    print(message)
    logger.error(message)
    write_to_txt_file(message)

async def main_loop(server_url, service_id, workspace, token):
    while True:
        await call_service_with_retries(server_url, service_id, "hello1", workspace, token)
        await call_service_with_retries(server_url, service_id, "hello2", workspace, token)
        # Add a delay between loops if needed
        await asyncio.sleep(10)

if __name__ == "__main__":
    server_url = "https://hypha.aicell.io"
    workspace = "reef-imaging"
    service_id = "hello-world"
    token = os.environ.get("REEF_WORKSPACE_TOKEN")
    asyncio.run(main_loop(server_url, service_id, workspace, token))