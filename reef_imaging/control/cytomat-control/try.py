from cytomat import Cytomat

# Initialize Cytomat with the correct serial port and configuration file path
c = Cytomat("/dev/ttyUSB0", json_path="/home/tao/workspace/cytomat-controller/docs/config.json")

# Initialize the plate handler
c.plate_handler.initialize()

print("Plate on transfer station?", c.overview_status.transfer_station_occupied)
print(c.overview_status)

slot = int(5)
c.wait_until_not_busy(timeout=50)
c.plate_handler.move_plate_from_transfer_station_to_slot(slot)
c.wait_until_not_busy(timeout=50)
c.plate_handler.move_plate_from_slot_to_transfer_station(slot)
c.wait_until_not_busy(timeout=50)