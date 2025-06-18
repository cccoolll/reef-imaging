# Real-Time Image Map with Live Stitching Feature

This feature adds real-time image stitching capabilities to the microscope control system, creating a live map of the microscope's field of view as it moves across the sample.

## Overview

The live stitching system creates a continuously updated zarr canvas that users can access through chunk-based queries. As the microscope moves and captures video frames, each frame is automatically placed in the correct location on a large canvas based on the stage position.

## Files Created/Modified

### New Files
- `stitcher.py` - Core stitching module with the following classes:
  - `ImagingParameters` - Handles pixel size calculations
  - `StitchCanvas` - Manages coordinate transformations
  - `ZarrCanvasManager` - Handles zarr file operations
  - `LiveStitcher` - Main coordination class

- `test_live_stitching.py` - Test script for the stitching functionality

### Modified Files
- `mirror_squid_control.py` - Integrated stitcher into the mirror service

## Key Features

### 1. Multi-Scale Canvas
- Creates zarr files with pyramid levels (scale1-5)
- scale1: Base resolution (750x750 frame resolution)
- scale2-5: Progressively downsampled versions (4x reduction per level)
- 256x256 pixel chunks for efficient access

### 2. Real-Time Frame Placement
- Extracts stage position from video frames
- Converts physical coordinates (mm) to pixel coordinates
- Places frames directly on canvas without complex blending
- Updates all pyramid levels automatically

### 3. Hypha Service Integration
- New service methods for canvas access:
  - `get_canvas_chunk(x_mm, y_mm, scale_level)` - Get chunk at stage location
  - `reset_canvas()` - Reset the canvas

## Configuration

### Environment Variables
```bash
# Enable/disable live canvas capability (must be true for stitching to work)
LIVE_CANVAS_ENABLED=true

# Storage path for canvas files
LIVE_CANVAS_STORAGE_PATH=/data/live_canvas
```

### Command Line Arguments
```bash
# Enable live stitching (disabled by default)
--stitch

# Example usage
python mirror_squid_control.py --local-service-id microscope-control-squid-2 --stitch
```

**Note**: Both `LIVE_CANVAS_ENABLED=true` AND `--stitch` flag must be set for live stitching to be active.

### Stage Limits
The system uses predefined stage limits for canvas calculation:
```python
stage_limits = {
    "x_positive": 120,    # mm
    "x_negative": 0,      # mm  
    "y_positive": 86,     # mm
    "y_negative": 0,      # mm
    "z_positive": 6       # mm
}
```

### Canvas Calculation
- Physical area: 120mm x 86mm
- Pixel size: ~1.85µm per pixel (calculated from imaging parameters)
- Canvas size: ~65,000 x 46,000 pixels
- Storage: ~3GB for full canvas with compression
- Canvas naming: Uses microscope service ID (e.g., `live_canvas_microscope-control-squid-1.zarr`)

## Usage

### Starting the Service
Live stitching requires both environment variable and command line flag:

```bash
# Start with stitching enabled
LIVE_CANVAS_ENABLED=true python mirror_squid_control.py --local-service-id microscope-control-squid-2 --stitch

# Start without stitching (default)
python mirror_squid_control.py --local-service-id microscope-control-squid-2

# All service arguments work together
python mirror_squid_control.py \
  --local-service-id microscope-control-squid-2 \
  --cloud-service-id my-custom-service \
  --stitch
```

The service will log the stitching status on startup:
- `"Initializing live stitcher..."` - Stitching enabled and starting
- `"Live canvas is disabled via environment variable"` - `LIVE_CANVAS_ENABLED=false`
- `"Live stitching is disabled via command line argument"` - Missing `--stitch` flag

### Accessing Canvas Data
```python
# Get a chunk at a specific stage location (256x256 pixels)
chunk = await service.get_canvas_chunk(x_mm=10.5, y_mm=15.2, scale_level=1)

# Reset the canvas (clears all data and starts fresh)
result = await service.reset_canvas()
```

### Data Format
Canvas chunk data is returned as base64-encoded PNG in the following format:
```python
{
    "data": str,                            # Base64-encoded PNG image data
    "format": "png_base64",                 # Always "png_base64"
    "scale_level": int,                     # Scale level used
    "stage_location": {"x_mm": float, "y_mm": float},  # Input stage coordinates
    "chunk_coordinates": {"chunk_x": int, "chunk_y": int}  # Computed chunk coordinates
}
```

Reset canvas returns:
```python
{
    "status": "success",                    # "success" or "error"
    "message": "Canvas reset successfully"  # Status message
}
```

## Technical Details

### Thread Safety
- Uses asyncio queues for frame processing
- Thread pool executor for zarr operations
- RLock for thread-safe zarr updates

### Performance
- Non-blocking frame addition to prevent video delays
- Queue-based processing with configurable queue size (100 frames)
- Efficient zarr chunking for fast access
- PNG compression with base64 encoding for data transfer
- Automatic RGB to grayscale conversion for reduced storage

### Memory Management
- Lazy loading of zarr chunks
- Frame queue size limits
- Efficient numpy operations
- Automatic cleanup on service shutdown

## Testing

To test the live stitching functionality:

```bash
cd reef_imaging/control/mirror-services

# Start the service with stitching enabled
LIVE_CANVAS_ENABLED=true python mirror_squid_control.py --local-service-id microscope-control-squid-2 --stitch

# In another terminal, you can test the service methods via Hypha
```

The service will create a zarr canvas and begin stitching frames as video streaming occurs.

## Integration with WebRTC

The system integrates seamlessly with the existing WebRTC video streaming:
- Each video frame automatically includes stage position metadata
- Frames are added to the stitcher asynchronously without affecting video performance
- Canvas updates happen in the background

## Error Handling

The system includes robust error handling:
- Graceful degradation when stage position is unavailable
- Automatic canvas boundary checking
- Queue overflow protection
- Service restart recovery
- Comprehensive logging at all levels

## Future Enhancements

Potential improvements:
- Channel-based stitching for multi-fluorescence imaging
- Intelligent overlap blending algorithms
- Real-time canvas compression optimization
- WebRTC streaming of canvas tiles
- Interactive canvas navigation interface 