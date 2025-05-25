import os
import cv2
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional, Union, Any
import shutil
import re

class ImagingParameters:
    """Class for handling imaging parameters"""
    
    def __init__(self, parameter_file: str = None, parameters: Dict = None):
        if parameter_file:
            self.parameters = self.load_from_file(parameter_file)
        elif parameters:
            self.parameters = parameters
        else:
            self.parameters = {}
    
    def load_from_file(self, parameter_file: str) -> Dict:
        """Load imaging parameters from a JSON file."""
        with open(parameter_file, "r") as f:
            parameters = json.load(f)
        return parameters
    
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
        except KeyError:
            raise ValueError("Missing required parameters for pixel size calculation.")

        pixel_size_xy = pixel_size_um / (magnification / (objective_tube_lens_mm / tube_lens_mm))
        # Apply manual adjustment factor
        pixel_size_xy *= adjustment_factor
        print(f"Pixel size: {pixel_size_xy} µm (with adjustment factor: {adjustment_factor})")
        return pixel_size_xy


class ImageProcessor:
    """Class for processing and manipulating images"""
    
    @staticmethod
    def rotate_flip_image(image, angle=0, flip=True):
        """Rotate an image by a specified angle and flip if required."""
        # If no rotation, just do the flip if needed
        if angle == 0:
            if flip:
                return cv2.flip(image, -1)  # Flip horizontally and vertically
            return image
            
        # Rotating with improved method to prevent black edges
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Use a border to avoid black edges
        border_size = int(max(height, width) * 0.1)  # 10% border padding
        padded_image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, 
                                          cv2.BORDER_REPLICATE)
        
        # Adjust rotation center for padded image
        padded_center = (center[0] + border_size, center[1] + border_size)
        padded_rotation_matrix = cv2.getRotationMatrix2D(padded_center, angle, 1.0)
        
        # Rotate padded image
        padded_height, padded_width = padded_image.shape[:2]
        rotated_padded = cv2.warpAffine(padded_image, padded_rotation_matrix, (padded_width, padded_height), 
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # Crop back to original size
        rotated = rotated_padded[border_size:border_size+height, border_size:border_size+width]
        
        # Apply flip if required
        if flip:
            rotated = cv2.flip(rotated, -1)  # Flip horizontally and vertically
            
        return rotated
    
    @staticmethod
    def normalize_to_8bit(image):
        """Normalize an image to 8-bit format."""
        if image.dtype != np.uint8:
            img_min = np.percentile(image, 1)
            img_max = np.percentile(image, 99)
            image = np.clip(image, img_min, img_max)
            image = ((image - img_min) * 255 / (img_max - img_min)).astype(np.uint8)
        return image
    
    @staticmethod
    def create_weight_mask(height, width, feather_percent=0.05):
        """Create a weight mask with higher values in the center and feathering at edges."""
        y, x = np.mgrid[0:height, 0:width]
        center_y, center_x = height // 2, width // 2
        
        # Distance from center
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = min(width, height) / 2
        
        # Smooth falloff that's mostly 1.0 in the center and drops off sharply near edges
        weight_mask = np.clip(1.0 - (dist_from_center / max_dist)**2, 0.01, 1.0)**2
        weight_mask = weight_mask.astype(np.float32)
        
        # Apply feathering at edges
        feather_width = int(min(width, height) * feather_percent)
        if feather_width > 0:
            # Create edge mask
            edge_mask = np.ones((height, width), dtype=np.float32)
            edge_mask[:feather_width, :] *= np.linspace(0, 1, feather_width)[:, np.newaxis]
            edge_mask[-feather_width:, :] *= np.linspace(1, 0, feather_width)[:, np.newaxis]
            edge_mask[:, :feather_width] *= np.linspace(0, 1, feather_width)[np.newaxis, :]
            edge_mask[:, -feather_width:] *= np.linspace(1, 0, feather_width)[np.newaxis, :]
            
            # Apply edge feathering
            weight_mask *= edge_mask
            
        return weight_mask


class StitchCanvas:
    """Class for creating and managing the stitching canvas"""
    
    def __init__(self, stage_limits: Dict[str, float], pixel_size_xy: float):
        """
        Initialize the stitching canvas with stage limits and pixel size.
        
        Args:
            stage_limits: Dictionary with x_positive, x_negative, y_positive, y_negative, z_positive values in mm
            pixel_size_xy: Pixel size in micrometers
        """
        self.stage_limits = stage_limits
        self.pixel_size_xy = pixel_size_xy
        self.canvas_width, self.canvas_height = self._calculate_canvas_size()
        
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


class ZarrWriter:
    """Class for creating and writing to OME-NGFF (zarr) files"""
    
    def __init__(self, 
                output_folder: str, 
                canvas_width: int, 
                canvas_height: int, 
                channels: List[str], 
                zarr_filename: str = None, 
                pyramid_levels: int = 7, 
                chunk_size: Tuple[int, int] = (256, 256)):
        """
        Initialize the zarr writer with canvas dimensions and other parameters.
        
        Args:
            output_folder: Directory where zarr files will be saved
            canvas_width: Width of the canvas in pixels
            canvas_height: Height of the canvas in pixels
            channels: List of channel names
            zarr_filename: Name of the zarr file (default: "stitched_images.zarr")
            pyramid_levels: Number of pyramid levels to create (default: 7)
            chunk_size: Size of chunks for zarr storage (default: (256, 256))
        """
        self.output_folder = output_folder
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.channels = channels
        self.zarr_filename = zarr_filename or "stitched_images.zarr"
        self.pyramid_levels = pyramid_levels
        self.chunk_size = chunk_size
        self.datasets = None
        self.root = None
        
    def create_zarr_file(self) -> Tuple[Any, Dict]:
        """Create an OME-NGFF (zarr) file with a fixed canvas size and pyramid levels."""
        os.makedirs(self.output_folder, exist_ok=True)
        
        zarr_path = os.path.join(self.output_folder, self.zarr_filename)
        
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=store)

        # Print initial canvas size
        print(f"Initial canvas size: {self.canvas_width}x{self.canvas_height}")

        # Create datasets for each channel
        datasets = {}
        for channel in self.channels:
            print(f"\nCreating dataset for channel: {channel}")
            group = root.create_group(channel)
            datasets[channel] = group

            # Create base resolution (scale0)
            print(f"scale0: {self.canvas_height}x{self.canvas_width}")
            datasets[channel].create_dataset(
                "scale0",
                shape=(self.canvas_height, self.canvas_width),
                chunks=self.chunk_size,
                dtype=np.uint8,
                compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE),
                dimension_separator='/'
            )

            # Create pyramid levels
            for level in range(1, self.pyramid_levels):
                scale_name = f"scale{level}"
                level_shape = (
                    max(1, self.canvas_height // (4 ** level)),
                    max(1, self.canvas_width // (4 ** level))
                )
                level_chunks = (
                    min(self.chunk_size[0], level_shape[0]),
                    min(self.chunk_size[1], level_shape[1])
                )

                print(f"{scale_name}: {level_shape} (chunks: {level_chunks})")

                datasets[channel].create_dataset(
                    scale_name,
                    shape=level_shape,
                    chunks=level_chunks,
                    dtype=np.uint8,
                    compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE),
                    dimension_separator='/'
                )

        self.root = root
        self.datasets = datasets
        return root, datasets
    
    def update_pyramid(self, channel: str, level: int, image: np.ndarray, x_start: int, y_start: int) -> None:
        """Update a specific pyramid level with the given image."""
        scale = 4 ** level
        scale_name = f"scale{level}"

        try:
            pyramid_dataset = self.datasets[channel][scale_name]

            # Calculate the scaled coordinates
            x_scaled = x_start // scale
            y_scaled = y_start // scale

            # Calculate target dimensions
            target_height = max(1, image.shape[0] // scale)
            target_width = max(1, image.shape[1] // scale)

            # Only proceed if the target dimensions are meaningful
            if target_height == 0 or target_width == 0:
                print(f"Skipping level {level} due to zero dimensions")
                return

            # Resize the image
            scaled_image = cv2.resize(
                image, 
                (target_width, target_height),
                interpolation=cv2.INTER_AREA
            )

            # Calculate the valid region to update
            end_y = min(y_scaled + scaled_image.shape[0], pyramid_dataset.shape[0])
            end_x = min(x_scaled + scaled_image.shape[1], pyramid_dataset.shape[1])

            # Only update if we have a valid region
            if end_y > y_scaled and end_x > x_scaled:
                # Clip the image if necessary
                if end_y - y_scaled != scaled_image.shape[0] or end_x - x_scaled != scaled_image.shape[1]:
                    scaled_image = scaled_image[:(end_y - y_scaled), :(end_x - x_scaled)]

                # Place the image (no combining with existing data)
                pyramid_dataset[y_scaled:end_y, x_scaled:end_x] = scaled_image
            else:
                print(f"Skipping update for level {level} due to out-of-bounds coordinates")

        except Exception as e:
            print(f"Error updating pyramid level {level}: {e}")
        finally:
            # Clean up
            if 'scaled_image' in locals():
                del scaled_image


class ImageFileParser:
    """Class for parsing image filenames"""
    
    @staticmethod
    def parse_image_filenames(image_folder: str) -> List[Dict]:
        """Parse image filenames to extract FOV information."""
        image_files = [f for f in os.listdir(image_folder) if f.endswith(".bmp")]
        image_info = []

        for image_file in image_files:
            # New filename style: Region_FOVidx_Zidx_ChannelName.bmp
            # Example: A3_0_0_BF_LED_matrix_full.bmp
            prefix_parts = image_file.split('_', 3)  # Split into Region, FOV, Z, Channel
            if len(prefix_parts) >= 4: # Ensure Region, FOV, Z, and Channel parts exist
                region = prefix_parts[0]
                try:
                    fov_idx = int(prefix_parts[1])
                    z_idx = int(prefix_parts[2]) # Assuming z_idx is also an integer
                except ValueError:
                    print(f"Warning: Could not parse FOV/Z index from filename: {image_file}. Skipping.")
                    continue
                
                channel_name = prefix_parts[3].rsplit('.', 1)[0]

                image_info.append({
                    "filepath": os.path.join(image_folder, image_file),
                    "region": region,
                    "fov_idx": fov_idx, # New field for FOV index
                    "z_idx": z_idx,    # Z index (was string, now int based on example A3_0_0)
                    "channel_name": channel_name
                })
            else:
                print(f"Warning: Filename {image_file} does not match expected format Region_FOV_Z_Channel. Skipping.")


        # Sort images: primary key fov_idx, secondary key z_idx.
        # This is an assumption; user might need to specify a different order if required.
        image_info.sort(key=lambda x: (x["fov_idx"], x["z_idx"]))
        return image_info


def add_chunk_metadata(zarr_path: str, chunk_metadata: Dict[str, Dict[str, Any]]) -> bool:
    """
    Add per-chunk metadata to an existing OME-Zarr dataset.
    
    Args:
        zarr_path: Path to the OME-Zarr dataset
        chunk_metadata: Dictionary containing metadata for each chunk.
                       Keys should be string representations of chunk coordinates (e.g., "(0, 0, 0)")
                       Values should be dictionaries containing metadata fields
    
    Returns:
        bool: True if metadata was added successfully, False otherwise
    
    Example:
        >>> chunk_metadata = {
        ...     "(0, 0, 0)": {"date": "2025-05-06", "sample": "A1", "location_mm": (10.5, 15.2), "coordinates": (0, 0), "notes": "Sample area 1"},
        ...     "(0, 1, 0)": {"date": "2025-05-06", "sample": "A2", "location_mm": (10.5, 20.3), "coordinates": (0, 1), "notes": "Sample area 2"}
        ... }
        >>> add_chunk_metadata("/path/to/dataset.zarr", chunk_metadata)
        True
    """
    try:
        # Validate zarr path
        if not os.path.exists(zarr_path) or not os.path.isdir(zarr_path):
            print(f"Error: Invalid zarr path: {zarr_path}")
            return False
            
        # Validate metadata format
        for chunk_coords, metadata in chunk_metadata.items():
            # Check if chunk coordinates format is valid (e.g., "(0, 0, 0)")
            if not re.match(r"^\(\d+(?:,\s*\d+)*\)$", chunk_coords):
                print(f"Error: Invalid chunk coordinate format: {chunk_coords}")
                return False
                
            # Check if metadata is a dictionary
            if not isinstance(metadata, dict):
                print(f"Error: Metadata for chunk {chunk_coords} must be a dictionary")
                return False

        # Path to chunk metadata file
        metadata_file = os.path.join(zarr_path, "chunk_metadata.json")
        
        # Load existing metadata if it exists
        existing_metadata = {}
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
                print(f"Found existing metadata for {len(existing_metadata)} chunks")
            except json.JSONDecodeError:
                print(f"Warning: Could not parse existing metadata file, creating new one")
            except Exception as e:
                print(f"Warning: Error reading existing metadata file: {e}")
        
        # Update metadata with new values
        for chunk_coords, metadata in chunk_metadata.items():
            if chunk_coords in existing_metadata:
                existing_metadata[chunk_coords].update(metadata)
            else:
                existing_metadata[chunk_coords] = metadata
                
        # Write updated metadata back to file
        with open(metadata_file, 'w') as f:
            json.dump(existing_metadata, f, indent=2)
            
        print(f"Successfully added metadata for {len(chunk_metadata)} chunks to {metadata_file}")
        return True
        
    except Exception as e:
        print(f"Error adding chunk metadata: {e}")
        import traceback
        traceback.print_exc()
        return False
        
class StitchManager:
    """Manages the image stitching process"""
    
    def __init__(self):
        """Initialize the stitch manager."""
        self.image_processor = ImageProcessor()
        self.zarr_writer = None
        self.canvas = None
        
    def setup_from_parameters(self, 
                            parameter_file: str, 
                            stage_limits: Dict[str, float], 
                            output_folder: str,
                            zarr_filename: str = None,
                            channels: List[str] = None,
                            pyramid_levels: int = 7) -> None:
        """
        Set up the stitching environment from imaging parameters.
        
        Args:
            parameter_file: Path to the imaging parameters JSON file
            stage_limits: Dictionary with stage limits in mm
            output_folder: Directory where zarr files will be saved
            zarr_filename: Name of the zarr file
            channels: List of channel names
            pyramid_levels: Number of pyramid levels
        """
        # Load imaging parameters
        imaging_params = ImagingParameters(parameter_file=parameter_file)
        pixel_size_xy = imaging_params.get_pixel_size()
        
        # if zarr file exists, delete it
        zarr_path = os.path.join(output_folder, zarr_filename)
        if os.path.exists(zarr_path):
            # Use shutil.rmtree instead of os.remove for directories
            shutil.rmtree(zarr_path)

        # Create canvas
        self.canvas = StitchCanvas(stage_limits, pixel_size_xy)
        
        # Set up zarr writer if channels are provided
        if channels:
            self.zarr_writer = ZarrWriter(
                output_folder,
                self.canvas.canvas_width,
                self.canvas.canvas_height,
                channels,
                zarr_filename=zarr_filename,
                pyramid_levels=pyramid_levels
            )
            _, self.datasets = self.zarr_writer.create_zarr_file()
    
    def load_existing_zarr(self, zarr_path: str) -> None:
        """
        Load an existing zarr file for appending.
        
        Args:
            zarr_path: Path to the existing zarr file
        """
        if os.path.exists(zarr_path):
            print(f"Loading existing zarr file: {zarr_path}")
            self.datasets = zarr.open(zarr_path, mode="a")
            return True
        return False
    
    def process_images(self, 
                     image_info: List[Dict], 
                     coordinates: pd.DataFrame, 
                     selected_channel: List[str] = None,
                     pyramid_levels: int = 7,
                     rotation_angle: float = 0.1, 
                     blend_overlap: bool = False,  # Default to False to avoid memory issues
                     feather_percent: float = 0.05,
                     blend_tile_size: int = 4096) -> None:
        """
        Process images and place them on the canvas based on physical coordinates.
        
        Args:
            image_info: List of dictionaries with image information
            coordinates: DataFrame with coordinates information
            selected_channel: List of channels to process
            pyramid_levels: Number of pyramid levels
            rotation_angle: Angle in degrees to rotate images before processing
            blend_overlap: Whether to use alpha blending for overlapping regions
            feather_percent: Percentage of image size to use for edge feathering
            blend_tile_size: Maximum size of tiles to use for local blending
        """
        if selected_channel is None:
            raise ValueError("selected_channel must be a list of channels to process.")

        # Get canvas dimensions from the dataset
        canvas_width = self.datasets[selected_channel[0]]["scale0"].shape[1]
        canvas_height = self.datasets[selected_channel[0]]["scale0"].shape[0]
        
        # Process images by position (region, fov_idx, z_idx) together
        position_groups = {}
        
        for info in image_info:
            # Use fov_idx and z_idx from parsed image filenames
            position_key = (info["region"], info["fov_idx"], info["z_idx"])
            if position_key not in position_groups:
                position_groups[position_key] = []
            position_groups[position_key].append(info)
        
        # Sort position keys: region (alphabetical), fov_idx (decreasing), z_idx (ascending)
        # This mimics the previous sort: region, -x_idx (fov_idx), y_idx (z_idx)
        sorted_positions = sorted(position_groups.keys(), key=lambda pos: (
            pos[0],   # region
            -pos[1],  # fov_idx decreasing
            pos[2]    # z_idx ascending
        ))
        
        # Find the first image to determine tile dimensions
        sample_image = None
        for position in sorted_positions:
            if len(position_groups[position]) > 0:
                sample_info = position_groups[position][0]
                sample_path = sample_info["filepath"]
                sample_image = cv2.imread(sample_path, cv2.IMREAD_ANYDEPTH)
                if sample_image is not None:
                    sample_image = self.image_processor.rotate_flip_image(sample_image, angle=rotation_angle)
                    break
        
        if sample_image is None:
            raise ValueError("Could not find any valid images to process")
        
        tile_height, tile_width = sample_image.shape[:2]
        print(f"Tile dimensions: {tile_width}x{tile_height}")
        
        # Process all positions
        for position in tqdm(sorted_positions, desc="Processing positions"):
            region, fov_idx, z_idx = position # Unpack according to new position_key

            # Get coordinates for this position using new column names from coordinates.csv
            matching_rows = coordinates[(
                (coordinates["region"] == region) & 
                (coordinates["fov"] == fov_idx) &       # Use 'fov' column from CSV
                (coordinates["z_level"] == z_idx)     # Use 'z_level' column from CSV
            )]
            if matching_rows.empty:
                print(f"Warning: No coordinates found for position {position} (region={region}, fov={fov_idx}, z_level={z_idx})")
                continue
                
            coord_row = matching_rows.iloc[0]
            x_mm, y_mm = coord_row["x (mm)"], coord_row["y (mm)"]
            
            # Convert physical coordinates to pixel coordinates
            x_start, y_start = self.canvas.physical_to_pixel_coordinates(x_mm, y_mm)

            # Process all channels at this position
            for info in position_groups[position]:
                channel = info["channel_name"]
                if channel not in selected_channel:
                    continue

                # Read and preprocess image
                image = cv2.imread(info["filepath"], cv2.IMREAD_ANYDEPTH)
                if image is None:
                    print(f"Error: Unable to read {info['filepath']}")
                    continue

                # Normalize image to 8-bit
                image = self.image_processor.normalize_to_8bit(image)
                
                # Rotate and flip with the specified angle
                image = self.image_processor.rotate_flip_image(image, angle=rotation_angle)
                
                # Check if image dimensions match the expected tile size
                if image.shape[0] != tile_height or image.shape[1] != tile_width:
                    # Create a new empty image with the expected tile size
                    temp_image = np.zeros((tile_height, tile_width), dtype=np.uint8)
                    # Copy as much of the original image as fits
                    h = min(image.shape[0], tile_height)
                    w = min(image.shape[1], tile_width)
                    temp_image[:h, :w] = image[:h, :w]
                    image = temp_image

                # Validate and clip the placement region
                if x_start < 0 or y_start < 0 or x_start >= canvas_width or y_start >= canvas_height:
                    print(f"Warning: Image at {info['filepath']} is out of canvas bounds.")
                    continue

                # Calculate end positions and clip image if necessary
                x_end = min(x_start + image.shape[1], canvas_width)
                y_end = min(y_start + image.shape[0], canvas_height)

                # Create a trimmed image if it exceeds canvas boundaries
                if x_end - x_start != image.shape[1] or y_end - y_start != image.shape[0]:
                    trimmed_image = image[:(y_end-y_start), :(x_end-x_start)]
                else:
                    trimmed_image = image
                    
                # Apply simplified edge blending without using full weight map
                if blend_overlap:
                    # Create a feathered edge mask
                    edge_mask = self.image_processor.create_weight_mask(
                        trimmed_image.shape[0], 
                        trimmed_image.shape[1], 
                        feather_percent
                    )
                    
                    # Read the current data from the zarr
                    current_data = self.datasets[channel]["scale0"][y_start:y_end, x_start:x_end]
                    
                    # Calculate blended image (use additive blending instead of multiplicative)
                    # This preserves brightness better at edges
                    blended_image = np.zeros_like(trimmed_image, dtype=np.float32)
                    
                    # Convert to float for blending
                    current_float = current_data.astype(np.float32)
                    trimmed_float = trimmed_image.astype(np.float32)
                    
                    # Blend using the edge mask
                    blended_image = (1.0 - edge_mask) * current_float + edge_mask * trimmed_float
                    
                    # Convert back to uint8
                    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
                    
                    # Place the blended image
                    self.datasets[channel]["scale0"][y_start:y_end, x_start:x_end] = blended_image
                else:
                    # Direct placement without blending
                    self.datasets[channel]["scale0"][y_start:y_end, x_start:x_end] = trimmed_image

                # Update pyramid levels
                for level in range(1, pyramid_levels):
                    self.zarr_writer.update_pyramid(channel, level, image, x_start, y_start)

                del image

    def add_dataset_metadata(self, zarr_path: str, metadata: Dict[str, Any]) -> bool:
        """
        Add general metadata to the root of an OME-Zarr dataset.
        
        Args:
            zarr_path: Path to the OME-Zarr dataset
            metadata: Dictionary containing dataset-level metadata
            
        Returns:
            bool: True if metadata was added successfully, False otherwise
        """
        try:
            # Validate zarr path
            if not os.path.exists(zarr_path) or not os.path.isdir(zarr_path):
                print(f"Error: Invalid zarr path: {zarr_path}")
                return False
                
            # Path to dataset metadata file
            metadata_file = os.path.join(zarr_path, "dataset_metadata.json")
            
            # Load existing metadata if it exists
            existing_metadata = {}
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        existing_metadata = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse existing metadata file, creating new one")
                except Exception as e:
                    print(f"Warning: Error reading existing metadata file: {e}")
            
            # Update metadata with new values
            existing_metadata.update(metadata)
                    
            # Write updated metadata back to file
            with open(metadata_file, 'w') as f:
                json.dump(existing_metadata, f, indent=2)
                
            print(f"Successfully added dataset metadata to {metadata_file}")
            return True
            
        except Exception as e:
            print(f"Error adding dataset metadata: {e}")
            import traceback
            traceback.print_exc()
            return False


def stitch_images_example() -> None:
    """Example of stitching images"""
    # Paths and parameters
    data_folder = "/media/reef/harddisk/20250429-scan-time-lapse_2025-04-29_15-38-36.107800"
    image_folder = os.path.join(data_folder, "0")
    parameter_file = os.path.join(data_folder, "acquisition parameters.json")
    coordinates_file = os.path.join(image_folder, "coordinates.csv")
    output_folder = "/media/reef/harddisk/test_stitch_zarr"
    
    # Extract date and time from the folder name
    folder_name = os.path.basename(data_folder)
    date_time_str = folder_name.split('_')[1] + '_' + folder_name.split('_')[2].split('.')[0]
    zarr_filename = f"{date_time_str}.zarr"
    
    # Load coordinates
    coordinates = pd.read_csv(coordinates_file)
    
    # Predefined stage limits
    stage_limits = {
        "x_positive": 120,
        "x_negative": 0,
        "y_positive": 86,
        "y_negative": 0,
        "z_positive": 6
    }
    
    # Parse image filenames and get unique channels
    image_parser = ImageFileParser()
    image_info = image_parser.parse_image_filenames(image_folder)
    channels = list(set(info["channel_name"] for info in image_info))
    selected_channel = ['BF_LED_matrix_full', 'Fluorescence_488_nm_Ex', 'Fluorescence_561_nm_Ex']
    
    # Initialize stitch manager
    stitch_manager = StitchManager()
    
    # Setup from parameters
    stitch_manager.setup_from_parameters(
        parameter_file=parameter_file,
        stage_limits=stage_limits,
        output_folder=output_folder,
        zarr_filename=zarr_filename,
        channels=selected_channel,
        pyramid_levels=6
    )
    
    # Process images
    stitch_manager.process_images(
        image_info=image_info,
        coordinates=coordinates,
        selected_channel=selected_channel,
        pyramid_levels=6
    )
    
    print(f"Stitching complete. Output saved to {os.path.join(output_folder, zarr_filename)}")

if __name__ == "__main__":
    stitch_images_example() 