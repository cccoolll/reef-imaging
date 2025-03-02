import asyncio
from hypha_rpc import connect_to_server
from dorna_controller_local import move_sample_from_microscope_to_incubator, move_sample_from_incubator_to_microscope, move_plate
async def start_server(server_url):
    server = await connect_to_server({"server_url": server_url})

    

    svc = await server.register_service({
        "name": "Robotic Arm Control",
        "id": "robotic-arm-control",
        "config": {
            "visibility": "public"
        },
        "methods": {
            "move_sample_from_microscope_to_incubator": move_sample_from_microscope_to_incubator,
            "move_sample_from_incubator_to_microscope": move_sample_from_incubator_to_microscope,
            "move_plate": move_plate

        }
    })

    print(f"Incubator control service registered at workspace: {server.config.workspace}, id: {svc.id}")

    print(f'You can use this service using the service id: {svc.id}')
    id = svc.id.split(":")[1]
    print(f'You can use this service using the service id: {id}')
    # Keep the server running
    await server.serve()

if __name__ == "__main__":
    server_url = "http://localhost:9527"
    asyncio.run(start_server(server_url))
