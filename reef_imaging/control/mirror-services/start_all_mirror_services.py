#!/usr/bin/env python3
import os
import subprocess
import argparse
import signal
import time
import sys

def start_service(service_script, service_name):
    """Start a mirror service in a subprocess"""
    print(f"Starting {service_name} mirror service...")
    process = subprocess.Popen(
        [sys.executable, service_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    return process

def stop_services(processes):
    """Stop all running mirror services"""
    for name, process in processes.items():
        print(f"Stopping {name} mirror service...")
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"  {name} service did not terminate gracefully, killing...")
            process.kill()

def monitor_processes(processes):
    """Monitor the running processes and print their output"""
    try:
        while True:
            all_terminated = True
            for name, process in processes.items():
                if process.poll() is None:  # Process is still running
                    all_terminated = False
                    # Print any new output
                    for line in iter(process.stdout.readline, ''):
                        if not line:
                            break
                        print(f"[{name}] {line.strip()}")
                else:
                    # Process terminated
                    if process.returncode != 0:
                        print(f"WARNING: {name} mirror service exited with code {process.returncode}")
                        # Restart the process
                        if name == "microscope":
                            processes[name] = start_service("mirror_squid_control.py", "Microscope")
                        elif name == "robotic_arm":
                            processes[name] = start_service("mirror_robotic_arm.py", "Robotic Arm")
                        elif name == "incubator":
                            processes[name] = start_service("mirror_incubator.py", "Incubator")
                        print(f"Restarted {name} mirror service")
                        all_terminated = False
            
            if all_terminated:
                print("All mirror services have terminated")
                break
                
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, stopping services...")
        stop_services(processes)
        print("All services stopped")

def main():
    """Start all mirror services"""
    parser = argparse.ArgumentParser(description="Start all REEF imaging mirror services")
    parser.add_argument("--microscope-only", action="store_true", help="Start only the microscope mirror service")
    parser.add_argument("--robotic-arm-only", action="store_true", help="Start only the robotic arm mirror service")
    parser.add_argument("--incubator-only", action="store_true", help="Start only the incubator mirror service")
    args = parser.parse_args()

    # Change to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    processes = {}

    # Start services based on arguments
    if args.microscope_only:
        processes["microscope"] = start_service("mirror_squid_control.py", "Microscope")
    elif args.robotic_arm_only:
        processes["robotic_arm"] = start_service("mirror_robotic_arm.py", "Robotic Arm")
    elif args.incubator_only:
        processes["incubator"] = start_service("mirror_incubator.py", "Incubator")
    else:
        # Start all services
        processes["microscope"] = start_service("mirror_squid_control.py", "Microscope")
        processes["robotic_arm"] = start_service("mirror_robotic_arm.py", "Robotic Arm")
        processes["incubator"] = start_service("mirror_incubator.py", "Incubator")

    print("All services started. Press Ctrl+C to stop.")
    monitor_processes(processes)

if __name__ == "__main__":
    main()