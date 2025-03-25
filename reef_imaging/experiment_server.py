import asyncio
from hypha_rpc import connect_to_server
import dotenv
import os
import logging

# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "experiment_server.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

async def start_server(server_url, workspace, token):
    server = await connect_to_server({"server_url": server_url, "workspace": workspace, "token": token})

    task_status = {
        "hello1": "not_started",
        "hello2": "not_started"
    }

    def hello1(name):
        task_status["hello1"] = "started"
        message = "Hello1 " + name
        print(message)
        logger.info(message)
        task_status["hello1"] = "finished"
        return message

    def hello2(name):
        task_status["hello2"] = "started"
        message = "Hello2 " + name
        print(message)
        logger.info(message)
        task_status["hello2"] = "finished"
        return message

    def get_status(task_name):
        return task_status.get(task_name, "unknown")

    def reset_status(task_name):
        task_status[task_name] = "not_started"

    svc = await server.register_service({
        "name": "Hello World",
        "id": "hello-world",
        "config": {
            "visibility": "public"
        },
        "hello1": hello1,
        "hello2": hello2,
        "get_status": get_status,
        "reset_status": reset_status
    })

    message = f"Hello world service registered at workspace: {server.config.workspace}, id: {svc.id}"
    print(message)
    logger.info(message)

    # Keep the server running
    await server.serve()

if __name__ == "__main__":
    dotenv.load_dotenv()
    server_url = "https://hypha.aicell.io"  # Change to your actual server URL if needed
    workspace = "reef-imaging"
    token = os.environ.get("REEF_WORKSPACE_TOKEN")
    asyncio.run(start_server(server_url, workspace, token))