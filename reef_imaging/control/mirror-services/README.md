# REEF Imaging Mirror Services

This directory contains mirror services that proxy requests from the cloud server (https://hypha.aicell.io) to the local REEF imaging services. These services allow remote control of the microscope, robotic arm, and incubator while maintaining local control and safety measures.

## Services Overview
![mirror sercies flow](docs/mirror_services_flow.png)
1. **Mirror Microscope Control Service** (`mirror_squid_control.py`)
   - Mirrors the local microscope control service
   - Handles imaging, stage movement, and microscope settings
   - Provides status monitoring and health checks

2. **Mirror Robotic Arm Service** (`mirror_robotic_arm.py`)
   - Mirrors the local robotic arm control service
   - Manages sample transfer between microscope and incubator
   - Provides movement control and status monitoring

3. **Mirror Incubator Service** (`mirror_incubator.py`)
   - Mirrors the local incubator control service
   - Handles sample storage and environmental control
   - Monitors temperature, CO2 levels, and sample status

## Prerequisites

- Python 3.7+
- `hypha_rpc` package
- `python-dotenv` package
- Access to both local and cloud Hypha servers

## Environment Setup

Create a `.env` file in the mirror-services directory with the following variables:

```env
REEF_WORKSPACE_TOKEN=your_cloud_token_here
REEF_LOCAL_TOKEN=your_local_token_here
MICROSCOPE_SERVICE_ID=microscope-control-squid-1
```

## Running the Services

### Starting All Services

To start all mirror services simultaneously:

```bash
./start_all_mirror_services.py
```

### Starting Individual Services

To start specific services:

```bash
./start_all_mirror_services.py --microscope-only    # Start only microscope service
./start_all_mirror_services.py --robotic-arm-only   # Start only robotic arm service
./start_all_mirror_services.py --incubator-only     # Start only incubator service
```

### Service Management

- The services will automatically reconnect if the connection is lost
- Health checks run every 30 seconds
- Press Ctrl+C to gracefully stop the services
- All operations are logged to service-specific log files

## Service URLs

- Cloud Server: https://hypha.aicell.io
- Local Server: http://reef.dyn.scilifelab.se:9527

## Service IDs

- Mirror Microscope Service: `mirror-squid-control`
- Mirror Robotic Arm Service: `mirror-robotic-arm-control`
- Mirror Incubator Service: `mirror-incubator-control`

## Logging

Each service maintains its own log file:
- `mirror_squid_control_service.log`
- `mirror_robotic_arm_service.log`
- `mirror_incubator_service.log`

Log files use rotating file handlers with a maximum size of 100KB and keep 3 backup files.

## Health Monitoring

All services include:
- Automatic reconnection to local services
- Task status tracking
- Error handling and logging
- Service health checks every 30 seconds

## Task Status Tracking

Each service maintains status information for all operations:
- `not_started`: Task hasn't been initiated
- `started`: Task is in progress
- `finished`: Task completed successfully
- `failed`: Task failed with an error

## Error Handling

- All operations include comprehensive error handling
- Errors are logged with full stack traces
- Services attempt to reconnect automatically on failure
- Failed operations return appropriate error messages

## Security

- All connections require authentication tokens
- Separate tokens for local and cloud access
- Environment variables used for sensitive information
- No direct hardware control - all operations proxy through local services 