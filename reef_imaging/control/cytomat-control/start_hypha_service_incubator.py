import asyncio
from hypha_rpc import connect_to_server, login
from cytomat import Cytomat

async def start_server(server_url):
    server = await connect_to_server({"server_url": server_url})
    
    def initialize():
        c = Cytomat("/dev/ttyUSB0", json_path="/home/tao/workspace/cytomat-controller/docs/config.json")
        c.plate_handler.initialize()
        
    def move_plate(slot):
        # Initialize Cytomat with the correct serial port and configuration file path
        c = Cytomat("/dev/ttyUSB0", json_path="/home/tao/workspace/cytomat-controller/docs/config.json")
        c.wait_until_not_busy(timeout=50)
        c.plate_handler.initialize()
        c.wait_until_not_busy(timeout=50)
        c.plate_handler.move_plate_from_transfer_station_to_slot(slot)
        c.wait_until_not_busy(timeout=50)
        c.plate_handler.move_plate_from_slot_to_transfer_station(slot)
        c.wait_until_not_busy(timeout=50)
        return f"Plate moved to slot {slot} and back to transfer station."
    
    def put_sample_from_transfer_station_to_slot(slot=5):
        c = Cytomat("/dev/ttyUSB0")
        c.plate_handler.move_plate_from_transfer_station_to_slot(slot)

    def get_sample_from_slot_to_transfer_station(slot=5):
        c = Cytomat("/dev/ttyUSB0")
        c.plate_handler.move_plate_from_slot_to_transfer_station(slot)
    
    def is_busy():
        c = Cytomat("/dev/ttyUSB0")
        return c.overview_status.busy
    

    svc = await server.register_service({
        "name": "Incubator Control",
        "id": "incubator-control",
        "config": {
            "visibility": "public"
        },
        "initialize": initialize,
        "put_sample_from_transfer_station_to_slot": put_sample_from_transfer_station_to_slot,
        "get_sample_from_slot_to_transfer_station": get_sample_from_slot_to_transfer_station,
        "is_busy": is_busy
    })

    print(f"Incubator control service registered at workspace: {server.config.workspace}, id: {svc.id}")

    print(f'You can use this service using the service id: {svc.id}')
    id = svc.id.split(":")[1]

    print(f"You can also test the service via the HTTP proxy: {server_url}/{server.config.workspace}/services/{id}/initialize")

    # Keep the server running
    await server.serve()

if __name__ == "__main__":
    server_url = "http://localhost:9527"
    asyncio.run(start_server(server_url))
