# Reef Imaging

A platform for automated microscope control, image acquisition, data management, and analysis for reef biological experiments.

## Key Components

### Orchestration

- **orchestrator.py** - Main orchestration system that manages plate handling, microscope control, and imaging sequences
- **orchestrator_simulation.py** - Simulation version of the orchestrator for testing

### Hardware Control

- **control/** - Hardware control modules
  - **dorna-control/** - Control for Dorna robotic arm
  - **cytomat-control/** - Control for Cytomat incubator
  - **squid-control/** - Control for SQUID microscope (includes built-in mirror functionality)
  - **mirror-services/** - Services for mirroring data (robotic arm and incubator only)

### Data Management

- **hypha_tools/** - Utilities for working with the Hypha platform for data management
  - Automated uploaders for treatment data and stitched images
  - Storage utilities
  - Artifact management tools

### Supplementary Tools

- **tools/** - Utilities and supplementary tools
  - **micro_sam/** - Micro-scale Segment Anything Model integration
  - **dorna_stress_test.py** - Stress testing for the Dorna robotic arm

- **lab_live_stream/** - Tools for lab camera streaming
  - **FYIR_camera.py** - FYIR camera interface
  - **realsense_camera.py** - RealSense camera interface

## Log Files

The system maintains rotating log files for the orchestrator and other components, with formats like:
- `orchestrator-[timestamp].log` - Main orchestrator logs
- `orchestrator-test-[timestamp].log` - Test orchestrator logs
- `uc2_fucci_time_lapse_scan.log` - Specific experiment logs

## Usage

To run the main orchestrator:

```bash
python orchestrator.py
```

For local development and testing:

```bash
python orchestrator_simulation.py --local
```

## Environment Setup

The system requires environment variables for authentication:

```
REEF_WORKSPACE_TOKEN=your_token_here
SQUID_WORKSPACE_TOKEN=your_token_here
```

For local development:
```
REEF_LOCAL_TOKEN=your_local_token
REEF_LOCAL_WORKSPACE=your_local_workspace
```

## Dependencies

- hypha_rpc - For RPC connections to the Hypha platform
- OpenCV - For image processing
- Pandas - For data management
- asyncio - For asynchronous operations
- dotenv - For environment variable management 