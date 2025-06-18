import os
import asyncio
import threading
import time
import json
from typing import Dict, Tuple, Optional, Any
import numpy as np
import cv2
import zarr
from numcodecs import Blosc
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class ImagingParameters:
    """Class for handling imaging parameters"""
    
    def __init__(self, parameters: Dict = None):
        self.parameters = parameters or {}
    
    def get_pixel_size(self, 
                     default_pixel_size: float = 1.85, 
                     default_tube_lens_mm: float = 50.0, 
                     default_objective_tube_lens_mm: float = 180.0, 
                     default_magnification: float = 20.0,
                     adjustment_factor: float = 0.936) -> float:
        """Calculate pixel size based on imaging parameters."""
        try:
            tube_lens_mm = float(self.parameters.get('tube_lens_mm', default_tube_lens_mm))
            pixel_size_um = float(self.parameters.get('sensor_pixel_size_um', default_pixel_size))
            objective_tube_lens_mm = float(self.parameters['objective'].get('tube_lens_f_mm', default_objective_tube_lens_mm))
            magnification = float(self.parameters['objective'].get('magnification', default_magnification))
        except (KeyError, TypeError):
            logger.warning("Missing parameters for pixel size calculation, using defaults")
            tube_lens_mm = default_tube_lens_mm
            pixel_size_um = default_pixel_size
            objective_tube_lens_mm = default_objective_tube_lens_mm
            magnification = default_magnification

        pixel_size_xy = pixel_size_um / (magnification / (objective_tube_lens_mm / tube_lens_mm))
        # Apply manual adjustment factor
        pixel_size_xy *= adjustment_factor
        logger.info(f"Pixel size: {pixel_size_xy} µm (with adjustment factor: {adjustment_factor})")
        return pixel_size_xy


class StitchCanvas:
    """Class for creating and managing the stitching canvas"""
    
    def __init__(self, stage_limits: Dict[str, float], pixel_size_xy: float, frame_size: Tuple[int, int] = (750, 750)):
        """
        Initialize the stitching canvas with stage limits and pixel size.
        
        Args:
            stage_limits: Dictionary with x_positive, x_negative, y_positive, y_negative values in mm
            pixel_size_xy: Pixel size in micrometers
            frame_size: Size of video frames (width, height) in pixels
        """
        self.stage_limits = stage_limits
        self.pixel_size_xy = pixel_size_xy
        self.frame_width, self.frame_height = frame_size
        self.canvas_width, self.canvas_height = self._calculate_canvas_size()
        
        logger.info(f"Canvas initialized: {self.canvas_width}x{self.canvas_height} pixels")
        logger.info(f"Frame size: {self.frame_width}x{self.frame_height} pixels")
        logger.info(f"Pixel size: {self.pixel_size_xy} µm")
        
    def _calculate_canvas_size(self) -> Tuple[int, int]:
        """Calculate the canvas size in pixels based on stage limits and pixel size."""
        x_range = (self.stage_limits["x_positive"] - self.stage_limits["x_negative"]) * 1000  # Convert mm to µm
        y_range = (self.stage_limits["y_positive"] - self.stage_limits["y_negative"]) * 1000  # Convert mm to µm
        canvas_width = int(x_range / self.pixel_size_xy)
        canvas_height = int(y_range / self.pixel_size_xy)
        return canvas_width, canvas_height
    
    def physical_to_pixel_coordinates(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
        """Convert physical coordinates (mm) to pixel coordinates on the canvas."""
        # Convert mm to μm and subtract stage offset
        x_um = (x_mm * 1000) - (self.stage_limits["x_negative"] * 1000)
        y_um = (y_mm * 1000) - (self.stage_limits["y_negative"] * 1000)
        
        # Convert to pixels
        x_pixel = int(x_um / self.pixel_size_xy)
        y_pixel = int(y_um / self.pixel_size_xy)
        
        return x_pixel, y_pixel


class ZarrCanvasManager:
    """Class for managing the zarr canvas file structure"""
    
    def __init__(self, 
                canvas_width: int, 
                canvas_height: int, 
                storage_path: str,
                chunk_size: Tuple[int, int] = (256, 256),
                pyramid_levels: int = 5):
        """
        Initialize the zarr canvas manager.
        
        Args:
            canvas_width: Width of the canvas in pixels
            canvas_height: Height of the canvas in pixels
            storage_path: Path where zarr file will be stored
            chunk_size: Size of chunks for zarr storage
            pyramid_levels: Number of pyramid levels (scale1-5)
        """
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.storage_path = storage_path
        self.chunk_size = chunk_size
        self.pyramid_levels = pyramid_levels
        self.zarr_store = None
        self.zarr_root = None
        self.datasets = {}
        
        # Thread safety
        self._write_lock = threading.RLock()
        
    def create_zarr_canvas(self) -> bool:
        """Create or open the zarr canvas with pyramid levels."""
        try:
            # Ensure storage directory exists
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Create zarr store and root group
            self.zarr_store = zarr.DirectoryStore(self.storage_path)
            
            # Check if zarr file already exists
            zarr_exists = os.path.exists(self.storage_path) and os.path.isdir(self.storage_path)
            
            if zarr_exists:
                # Open existing zarr file
                logger.info(f"Opening existing zarr canvas at: {self.storage_path}")
                self.zarr_root = zarr.open_group(store=self.zarr_store, mode='r+')
                
                # Load existing datasets
                for level in range(1, self.pyramid_levels + 1):
                    scale_name = f"scale{level}"
                    if scale_name in self.zarr_root:
                        self.datasets[scale_name] = self.zarr_root[scale_name]
                        logger.info(f"Loaded existing {scale_name}: {self.datasets[scale_name].shape}")
                    else:
                        logger.warning(f"Scale {scale_name} not found in existing zarr file")
                
                logger.info(f"Zarr canvas opened successfully with {len(self.datasets)} scales")
                return True
            else:
                # Create new zarr file
                logger.info(f"Creating new zarr canvas at: {self.storage_path}")
                logger.info(f"Canvas size: {self.canvas_width}x{self.canvas_height}")
                
                self.zarr_root = zarr.group(store=self.zarr_store)
                
                # Create datasets for each pyramid level (scale1-5)
                for level in range(1, self.pyramid_levels + 1):
                    scale_name = f"scale{level}"
                    scale_factor = 4 ** (level - 1)  # scale1=1x, scale2=4x, scale3=16x, etc.
                    
                    level_width = max(1, self.canvas_width // scale_factor)
                    level_height = max(1, self.canvas_height // scale_factor)
                    
                    level_chunks = (
                        min(self.chunk_size[0], level_height),
                        min(self.chunk_size[1], level_width)
                    )
                    
                    logger.info(f"Creating {scale_name}: {level_height}x{level_width} (chunks: {level_chunks})")
                    
                    self.datasets[scale_name] = self.zarr_root.create_dataset(
                        scale_name,
                        shape=(level_height, level_width),
                        chunks=level_chunks,
                        dtype=np.uint8,
                        compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
                        dimension_separator='/',
                        fill_value=0
                    )
                
                # Create metadata
                self._create_metadata()
                
                logger.info("Zarr canvas created successfully")
                return True
            
        except Exception as e:
            logger.error(f"Error creating zarr canvas: {e}")
            return False
    
    def _create_metadata(self):
        """Create metadata for the zarr canvas."""
        metadata_path = os.path.join(self.storage_path, "canvas_metadata.json")
        
        # Check if metadata already exists
        if os.path.exists(metadata_path):
            logger.info("Metadata file already exists, skipping creation")
            return
        
        metadata = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "chunk_size": self.chunk_size,
            "pyramid_levels": self.pyramid_levels,
            "description": "Live stitched microscope canvas"
        }
        
        # Save metadata as JSON file
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def update_canvas_region(self, scale_name: str, x_start: int, y_start: int, frame: np.ndarray) -> bool:
        """Update a region of the canvas with a new frame."""
        try:
            with self._write_lock:
                if scale_name not in self.datasets:
                    logger.error(f"Scale {scale_name} not found in datasets")
                    return False
                
                dataset = self.datasets[scale_name]
                
                # Calculate end coordinates
                x_end = min(x_start + frame.shape[1], dataset.shape[1])
                y_end = min(y_start + frame.shape[0], dataset.shape[0])
                
                # Validate coordinates
                if x_start >= dataset.shape[1] or y_start >= dataset.shape[0] or x_end <= x_start or y_end <= y_start:
                    logger.warning(f"Invalid coordinates for {scale_name}: ({x_start}, {y_start}) to ({x_end}, {y_end})")
                    return False
                
                # Trim frame if necessary
                frame_height = y_end - y_start
                frame_width = x_end - x_start
                
                if frame.shape[0] != frame_height or frame.shape[1] != frame_width:
                    frame = frame[:frame_height, :frame_width]
                
                # Update the dataset
                dataset[y_start:y_end, x_start:x_end] = frame
                
                return True
                
        except Exception as e:
            logger.error(f"Error updating canvas region: {e}")
            return False
    
    def get_canvas_chunk(self, scale_name: str, chunk_x: int, chunk_y: int) -> Optional[np.ndarray]:
        """Get a specific chunk from the canvas."""
        try:
            if scale_name not in self.datasets:
                logger.error(f"Scale {scale_name} not found in datasets")
                return None
                
            dataset = self.datasets[scale_name]
            
            # Calculate chunk boundaries
            y_start = chunk_y * self.chunk_size[0]
            y_end = min(y_start + self.chunk_size[0], dataset.shape[0])
            x_start = chunk_x * self.chunk_size[1]
            x_end = min(x_start + self.chunk_size[1], dataset.shape[1])
            
            # Validate boundaries
            if y_start >= dataset.shape[0] or x_start >= dataset.shape[1]:
                return None
            
            return dataset[y_start:y_end, x_start:x_end]
            
        except Exception as e:
            logger.error(f"Error getting canvas chunk: {e}")
            return None
    
    def get_canvas_region(self, scale_name: str, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
        """Get an arbitrary rectangular region from the canvas."""
        try:
            if scale_name not in self.datasets:
                logger.error(f"Scale {scale_name} not found in datasets")
                return None
                
            dataset = self.datasets[scale_name]
            
            # Validate and clip coordinates
            x1 = max(0, min(x1, dataset.shape[1] - 1))
            y1 = max(0, min(y1, dataset.shape[0] - 1))
            x2 = max(x1 + 1, min(x2, dataset.shape[1]))
            y2 = max(y1 + 1, min(y2, dataset.shape[0]))
            
            return dataset[y1:y2, x1:x2]
            
        except Exception as e:
            logger.error(f"Error getting canvas region: {e}")
            return None
    
    def get_canvas_info(self) -> Dict[str, Any]:
        """Get information about the canvas."""
        return {
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "chunk_size": self.chunk_size,
            "pyramid_levels": self.pyramid_levels,
            "scales": list(self.datasets.keys()),
            "storage_path": self.storage_path
        }


class LiveStitcher:
    """Main stitcher class that coordinates real-time stitching"""
    
    def __init__(self, 
                stage_limits: Dict[str, float],
                storage_path: str,
                imaging_parameters: Dict = None,
                frame_size: Tuple[int, int] = (750, 750),
                chunk_size: Tuple[int, int] = (256, 256),
                pyramid_levels: int = 5):
        """
        Initialize the live stitcher.
        
        Args:
            stage_limits: Dictionary with stage limits in mm
            storage_path: Path where zarr canvas will be stored
            imaging_parameters: Dictionary with imaging parameters
            frame_size: Size of video frames (width, height)
            chunk_size: Size of zarr chunks
            pyramid_levels: Number of pyramid levels
        """
        self.stage_limits = stage_limits
        self.storage_path = storage_path
        self.frame_size = frame_size
        self.chunk_size = chunk_size
        self.pyramid_levels = pyramid_levels
        
        # Initialize imaging parameters
        self.imaging_params = ImagingParameters(imaging_parameters)
        self.pixel_size_xy = self.imaging_params.get_pixel_size()
        
        # Initialize canvas
        self.canvas = StitchCanvas(stage_limits, self.pixel_size_xy, frame_size)
        
        # Initialize zarr manager
        self.zarr_manager = ZarrCanvasManager(
            self.canvas.canvas_width,
            self.canvas.canvas_height,
            storage_path,
            chunk_size,
            pyramid_levels
        )
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stitcher")
        self.frame_queue = asyncio.Queue(maxsize=100)
        self.running = False
        self.stitching_task = None
        
        logger.info("LiveStitcher initialized")
    
    async def start(self) -> bool:
        """Start the stitcher."""
        try:
            # Create zarr canvas
            success = await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                self.zarr_manager.create_zarr_canvas
            )
            
            if not success:
                logger.error("Failed to create zarr canvas")
                return False
            
            # Start stitching task
            self.running = True
            self.stitching_task = asyncio.create_task(self._stitching_loop())
            
            logger.info("LiveStitcher started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting stitcher: {e}")
            return False
    
    async def stop(self):
        """Stop the stitcher."""
        self.running = False
        
        if self.stitching_task:
            self.stitching_task.cancel()
            try:
                await self.stitching_task
            except asyncio.CancelledError:
                pass
        
        self.executor.shutdown(wait=True)
        logger.info("LiveStitcher stopped")
    
    async def add_frame(self, frame: np.ndarray, metadata: Dict[str, Any]):
        """Add a frame to the stitching queue."""
        try:
            if not self.running:
                return
            
            # Extract stage position from metadata
            if 'stage_position' not in metadata:
                logger.warning("No stage position in frame metadata")
                return
            
            stage_pos = metadata['stage_position']
            if 'x_mm' not in stage_pos or 'y_mm' not in stage_pos:
                logger.warning("Incomplete stage position in metadata")
                return
            
            frame_data = {
                'frame': frame,
                'x_mm': stage_pos['x_mm'],
                'y_mm': stage_pos['y_mm'],
                'timestamp': metadata.get('timestamp', time.time())
            }
            
            # Add to queue (non-blocking)
            try:
                self.frame_queue.put_nowait(frame_data)
            except asyncio.QueueFull:
                logger.warning("Frame queue full, dropping frame")
                
        except Exception as e:
            logger.error(f"Error adding frame: {e}")
    
    async def _stitching_loop(self):
        """Main stitching loop that processes frames from the queue."""
        logger.info("Stitching loop started")
        
        while self.running:
            try:
                # Get frame from queue with timeout
                frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)
                
                # Process frame in thread pool
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._process_frame,
                    frame_data
                )
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stitching loop: {e}")
        
        logger.info("Stitching loop stopped")
    
    def _process_frame(self, frame_data: Dict[str, Any]):
        """Process a single frame and update the canvas."""
        try:
            frame = frame_data['frame']
            x_mm = frame_data['x_mm']
            y_mm = frame_data['y_mm']
            
            # Convert physical coordinates to pixel coordinates
            x_pixel, y_pixel = self.canvas.physical_to_pixel_coordinates(x_mm, y_mm)
            
            logger.debug(f"Placing frame at physical ({x_mm}, {y_mm}) mm -> pixel ({x_pixel}, {y_pixel})")
            
            # Convert to grayscale if it's a color image
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert RGB to grayscale using standard weights
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                logger.debug(f"Converted RGB frame to grayscale: {frame.shape}")
            elif len(frame.shape) == 3 and frame.shape[2] == 1:
                # Remove single channel dimension
                frame = frame.squeeze(axis=2)
                logger.debug(f"Squeezed single channel frame: {frame.shape}")
            
            # Ensure frame is 8-bit
            if frame.dtype != np.uint8:
                # Simple normalization
                frame = ((frame - frame.min()) * 255 / (frame.max() - frame.min())).astype(np.uint8)
            
            # Update scale1 (base level)
            success = self.zarr_manager.update_canvas_region("scale1", x_pixel, y_pixel, frame)
            if not success:
                logger.warning(f"Failed to update scale1 at ({x_pixel}, {y_pixel})")
                return
            
            # Update pyramid levels
            self._update_pyramid_levels(frame, x_pixel, y_pixel)
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    def _update_pyramid_levels(self, frame: np.ndarray, x_pixel: int, y_pixel: int):
        """Update pyramid levels for the placed frame."""
        try:
            # Update scales 2-5
            for level in range(2, self.pyramid_levels + 1):
                scale_name = f"scale{level}"
                scale_factor = 4 ** (level - 1)
                
                # Calculate scaled coordinates
                scaled_x = x_pixel // scale_factor
                scaled_y = y_pixel // scale_factor
                
                # Calculate scaled frame size
                scaled_width = max(1, frame.shape[1] // scale_factor)
                scaled_height = max(1, frame.shape[0] // scale_factor)
                
                # Resize frame
                if scaled_width > 0 and scaled_height > 0:
                    scaled_frame = cv2.resize(
                        frame,
                        (scaled_width, scaled_height),
                        interpolation=cv2.INTER_AREA
                    )
                    
                    # Ensure scaled frame is also grayscale and 8-bit
                    if len(scaled_frame.shape) == 3:
                        if scaled_frame.shape[2] == 3:
                            scaled_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_RGB2GRAY)
                        elif scaled_frame.shape[2] == 1:
                            scaled_frame = scaled_frame.squeeze(axis=2)
                    
                    if scaled_frame.dtype != np.uint8:
                        scaled_frame = scaled_frame.astype(np.uint8)
                    
                    # Update canvas
                    self.zarr_manager.update_canvas_region(scale_name, scaled_x, scaled_y, scaled_frame)
                
        except Exception as e:
            logger.error(f"Error updating pyramid levels: {e}")
    
    # Service methods
    async def get_canvas_info(self) -> Dict[str, Any]:
        """Get canvas information."""
        base_info = self.zarr_manager.get_canvas_info()
        base_info.update({
            "pixel_size_um": self.pixel_size_xy,
            "stage_limits": self.stage_limits,
            "frame_size": self.frame_size
        })
        return base_info
    
    async def get_canvas_chunk(self, scale_level: int, chunk_x: int, chunk_y: int) -> Optional[np.ndarray]:
        """Get a canvas chunk as numpy array."""
        try:
            scale_name = f"scale{scale_level}"
            
            chunk = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.zarr_manager.get_canvas_chunk,
                scale_name, chunk_x, chunk_y
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error getting canvas chunk: {e}")
            return None
    
    async def get_canvas_region(self, scale_level: int, x1: int, y1: int, x2: int, y2: int) -> Optional[bytes]:
        """Get a canvas region as JPEG bytes."""
        try:
            scale_name = f"scale{scale_level}"
            
            region = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.zarr_manager.get_canvas_region,
                scale_name, x1, y1, x2, y2
            )
            
            if region is None:
                return None
            
            # Convert to JPEG bytes
            _, jpeg_data = cv2.imencode('.jpg', region, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return jpeg_data.tobytes()
            
        except Exception as e:
            logger.error(f"Error getting canvas region: {e}")
            return None
    
    async def get_canvas_overview(self, max_size: int = 1024) -> Optional[bytes]:
        """Get a heavily downsampled overview of the full canvas."""
        try:
            # Use the highest scale level available
            scale_name = f"scale{self.pyramid_levels}"
            
            if scale_name not in self.zarr_manager.datasets:
                return None
            
            dataset = self.zarr_manager.datasets[scale_name]
            overview = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: dataset[:]
            )
            
            # Further downscale if needed
            h, w = overview.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                overview = cv2.resize(overview, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Convert to JPEG bytes
            _, jpeg_data = cv2.imencode('.jpg', overview, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return jpeg_data.tobytes()
            
        except Exception as e:
            logger.error(f"Error getting canvas overview: {e}")
            return None 