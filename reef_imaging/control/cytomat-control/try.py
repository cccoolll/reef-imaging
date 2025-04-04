from cytomat import Cytomat

# Initialize Cytomat with the correct serial port and configuration file path
c = Cytomat("/dev/ttyUSB1", json_path="/home/tao/workspace/cytomat-controller/docs/config.json")

# Initialize the plate handler
#c.plate_handler.initialize()
print("Current temperature:", c.climate_controller.current_temperature)
#print("Current CO2 level:", c.climate_controller.current_co2)
#print("Plate on transfer station?", c.overview_status.transfer_station_occupied)
#print(c.overview_status)

slot = int(10)
# c.wait_until_not_busy(timeout=50)
c.plate_handler.move_plate_from_transfer_station_to_slot(slot)
# c.wait_until_not_busy(timeout=50)
# c.plate_handler.move_plate_from_slot_to_transfer_station(slot)
# c.wait_until_not_busy(timeout=50)