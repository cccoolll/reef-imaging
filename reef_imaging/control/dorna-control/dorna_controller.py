from dorna2 import Dorna
import time

class DornaController:
    def __init__(self, ip="192.168.2.20"):
        self.robot = Dorna()
        self.ip = ip

    def connect(self):
        self.robot.connect(self.ip)
        print("Connected to robot")

    def disconnect(self):
        self.robot.close()
        print("Disconnected from robot")

    def set_motor(self, state):
        self.robot.set_motor(state)

    def play_script(self, script_path, timeout=160):
        print("Playing script")
        self.robot.play_script(script_path, timeout=timeout)

    def is_busy(self):
        status = self.robot.track_cmd()
        print(f"Robot status: {status}")
        return status["union"].get("stat", -1) != 2

    def move_sample_from_microscope1_to_incubator(self, timeout=160):
        self.connect()
        self.set_motor(1)
        self.play_script("paths/microscope1_to_incubator.txt")

    def move_sample_from_incubator_to_microscope1(self, timeout=160):
        self.connect()
        self.set_motor(1)
        self.play_script("paths/incubator_to_microscope1.txt")
        self.disconnect()

    def move_plate(self, source, destination, timeout=160):
        if source == "microscope" and destination == "incubator":
            self.move_sample_from_microscope1_to_incubator()
        elif source == "incubator" and destination == "microscope1":
            self.move_sample_from_incubator_to_microscope1()
        else:
            print(f"Invalid source-destination combination: {source} to {destination}")

if __name__ == "__main__":
    controller = DornaController()
    # Example usage
    controller.connect()
    #move_plate(controller, "microscope1", "incubator")
    print("Is robot busy?", controller.is_busy())
    controller.disconnect()
