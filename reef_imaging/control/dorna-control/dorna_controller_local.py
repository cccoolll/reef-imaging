from dorna2 import Dorna
import time

def move_sample_from_microscope_to_incubator(timeout=160):
    robot = Dorna()
    
    robot.connect("192.168.2.20")
    print("Connected to robot")
    robot.set_motor(1)
    # Add your robotic arm movement script here
    print("Playing script")
    robot.play_script("paths/microscope_to_incubator.txt", timeout=timeout)
    robot.close()
    print("Disconnected from robot")

def move_sample_from_incubator_to_microscope(timeout=160):
    robot = Dorna()
    robot.connect("192.168.2.20")
    print("Connected to robot")
    robot.set_motor(1)
    # Add your robotic arm movement script here
    print("Playing script")
    robot.play_script("paths/incubator_to_microscope.txt",timeout=timeout)
    robot.close()
    print("Disconnected from robot")
