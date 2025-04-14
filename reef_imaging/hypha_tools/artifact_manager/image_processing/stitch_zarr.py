import os
import cv2
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from tqdm import tqdm
import json

def load_imaging_parameters(parameter_file):
    """Load imaging parameters from a JSON file."""
    with open(parameter_file, "r") as f:
        parameters = json.load(f)
    return parameters

def rotate_flip_image(image, angle=0, flip=True):
    """Rotate an image by a specified angle and flip if required."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    if flip:
      rotated = cv2.flip(rotated, -1) # Flip horizontally and vertically
    return rotated

def get_pixel_size(parameters, default_pixel_size=1.85, default_tube_lens_mm=50.0, default_objective_tube_lens_mm=180.0, default_magnification=20.0):
    """Calculate pixel size based on imaging parameters."""
    try:
        tube_lens_mm = float(parameters.get('tube_lens_mm', default_tube_lens_mm))
        pixel_size_um = float(parameters.get('sensor_pixel_size_um', default_pixel_size))
        objective_tube_lens_mm = float(parameters['objective'].get('tube_lens_f_mm', default_objective_tube_lens_mm))
        magnification = float(parameters['objective'].get('magnification', default_magnification))
        # Get manual adjustment factor, default to 1.0 (no adjustment)
        adjustment_factor = 0.936
    except KeyError:
        raise ValueError("Missing required parameters for pixel size calculation.")

    pixel_size_xy = pixel_size_um / (magnification / (objective_tube_lens_mm / tube_lens_mm))
    # Apply manual adjustment factor
    pixel_size_xy *= adjustment_factor
    print(f"Pixel size: {pixel_size_xy} µm (with adjustment factor: {adjustment_factor})")
    return pixel_size_xy

def create_canvas_size(stage_limits, pixel_size_xy):
    """Calculate the canvas size in pixels based on stage limits and pixel size."""
    x_range = (stage_limits["x_positive"] - stage_limits["x_negative"]) * 1000  # Convert mm to µm
    y_range = (stage_limits["y_positive"] - stage_limits["y_negative"]) * 1000  # Convert mm to µm
    canvas_width = int(x_range / pixel_size_xy)
    canvas_height = int(y_range / pixel_size_xy)
    return canvas_width, canvas_height

def parse_image_filenames(image_folder):
    """Parse image filenames to extract FOV information."""
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".bmp")]
    image_info = []

    for image_file in image_files:
        # Split only for the first 4 parts (region, x, y, z)
        prefix_parts = image_file.split('_', 4)  # Split into max 5 parts
        if len(prefix_parts) >= 5:
            region, x_idx, y_idx, z_idx = prefix_parts[:4]
            # Get the channel name by removing the extension
            channel_name = prefix_parts[4].rsplit('.', 1)[0]

            # Don't try to convert z_idx to int since it might be 'focus' or other non-numeric value
            image_info.append({
                "filepath": os.path.join(image_folder, image_file),
                "region": region,
                "x_idx": int(x_idx),
                "y_idx": int(y_idx),
                "z_idx": z_idx,  # Keep as string, no conversion to int
                "channel_name": channel_name
            })

    # Sort images according to the grid pattern described in stitch.py:
    # Upper left is (I,0), upper right (I,J), bottom left (0,0), bottom right (0,J)
    # This means I increases as you go up, J increases as you go right
    image_info.sort(key=lambda x: (-x["x_idx"], x["y_idx"]))
    return image_info

def create_ome_ngff(output_folder, canvas_width, canvas_height, channels, zarr_filename=None, pyramid_levels=7, chunk_size=(2048, 2048)):
    """Create an OME-NGFF (zarr) file with a fixed canvas size and pyramid levels."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Use provided zarr_filename or default to "stitched_images.zarr"
    zarr_filename = zarr_filename or "stitched_images.zarr"
    zarr_path = os.path.join(output_folder, zarr_filename)
    
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)

    # Print initial canvas size
    print(f"Initial canvas size: {canvas_width}x{canvas_height}")

    # Create datasets for each channel
    datasets = {}
    for channel in channels:
        print(f"\nCreating dataset for channel: {channel}")
        group = root.create_group(channel)
        datasets[channel] = group

        # Create base resolution (scale0)
        print(f"scale0: {canvas_height}x{canvas_width}")
        datasets[channel].create_dataset(
            "scale0",
            shape=(canvas_height, canvas_width),
            chunks=chunk_size,
            dtype=np.uint8,
            compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)
        )

        # Create pyramid levels
        for level in range(1, pyramid_levels):
            scale_name = f"scale{level}"
            level_shape = (
                max(1, canvas_height // (4 ** level)),
                max(1, canvas_width // (4 ** level))
            )
            level_chunks = (
                min(chunk_size[0], level_shape[0]),
                min(chunk_size[1], level_shape[1])
            )

            print(f"{scale_name}: {level_shape} (chunks: {level_chunks})")

            datasets[channel].create_dataset(
                scale_name,
                shape=level_shape,
                chunks=level_chunks,
                dtype=np.uint8,
                compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)
            )

    return root, datasets

def update_pyramid(datasets, channel, level, image, x_start, y_start):
    """Update a specific pyramid level with the given image."""
    scale = 4 ** level
    scale_name = f"scale{level}"

    try:
        pyramid_dataset = datasets[channel][scale_name]

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

def process_images(image_info, coordinates, datasets, pixel_size_xy, stage_limits, pyramid_levels=7, selected_channel=None):
    """
    Process images and place them on the canvas based on physical coordinates.
    No registration is performed - images are placed directly according to stage coordinates.
    """
    if selected_channel is None:
        raise ValueError("selected_channel must be a list of channels to process.")

    # Convert stage limits from mm to μm for offset calculation
    x_offset = stage_limits["x_negative"] * 1000  # Convert mm to µm
    y_offset = stage_limits["y_negative"] * 1000  # Convert mm to µm

    # Get canvas dimensions from the dataset
    canvas_width = datasets[selected_channel[0]]["scale0"].shape[1]
    canvas_height = datasets[selected_channel[0]]["scale0"].shape[0]
    
    # Process images by position (same x,y,region) together
    position_groups = {}
    
    for info in image_info:
        position_key = (info["region"], info["x_idx"], info["y_idx"])
        if position_key not in position_groups:
            position_groups[position_key] = []
        position_groups[position_key].append(info)
    
    # Sort position keys according to the grid pattern described in stitch.py:
    # Upper left is (I,0), upper right (I,J), bottom left (0,0), bottom right (0,J)
    # This means I increases as you go up, J increases as you go right
    sorted_positions = sorted(position_groups.keys(), key=lambda pos: (
        pos[0],  # region
        -pos[1],  # i decreasing (False in sort_values)
        pos[2]    # j ascending (True in sort_values)
    ))
    
    # Find the first image to determine tile dimensions
    sample_image = None
    for position in sorted_positions:
        if len(position_groups[position]) > 0:
            sample_info = position_groups[position][0]
            sample_path = sample_info["filepath"]
            sample_image = cv2.imread(sample_path, cv2.IMREAD_ANYDEPTH)
            if sample_image is not None:
                sample_image = rotate_flip_image(sample_image)
                break
    
    if sample_image is None:
        raise ValueError("Could not find any valid images to process")
    
    tile_height, tile_width = sample_image.shape[:2]
    print(f"Tile dimensions: {tile_width}x{tile_height}")
    
    # Process all positions
    for position in tqdm(sorted_positions, desc="Processing positions"):
        region, x_idx, y_idx = position

        # Get coordinates for this position
        matching_rows = coordinates[(coordinates["region"] == region) & 
                                   (coordinates["i"] == x_idx) & 
                                   (coordinates["j"] == y_idx)]
        if matching_rows.empty:
            print(f"Warning: No coordinates found for position {position}")
            continue
            
        coord_row = matching_rows.iloc[0]
        x_mm, y_mm = coord_row["x (mm)"], coord_row["y (mm)"]
        
        # Convert mm to μm, then to pixels
        x_um = (x_mm * 1000) - x_offset
        y_um = (y_mm * 1000) - y_offset
        x_start = int(x_um / pixel_size_xy)
        y_start = int(y_um / pixel_size_xy)

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
            if image.dtype != np.uint8:
                img_min = np.percentile(image, 1)
                img_max = np.percentile(image, 99)
                image = np.clip(image, img_min, img_max)
                image = ((image - img_min) * 255 / (img_max - img_min)).astype(np.uint8)
            
            # Rotate and flip
            image = rotate_flip_image(image)
            
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

            # Place image directly on canvas at base level (scale0)
            datasets[channel]["scale0"][y_start:y_end, x_start:x_end] = trimmed_image

            # Update pyramid levels
            for level in range(1, pyramid_levels):
                update_pyramid(datasets, channel, level, image, x_start, y_start)

            del image

def generate_coordinates_file(image_folder, config, wellplate_format):
    """Generate coordinates file for 96-well plate sample data."""
    wells = [f"{row}{col}" for row in "ABCDEFGH" for col in range(1, 13)]
    coordinates = []

    for well in wells:
        well_x = wellplate_format.A1_X_MM + (int(well[1:]) - 1) * wellplate_format.WELL_SPACING_MM
        well_y = wellplate_format.A1_Y_MM + (ord(well[0]) - ord('A')) * wellplate_format.WELL_SPACING_MM

        for i in range(config["Nx"]):
            for j in range(config["Ny"]):
                x_mm = well_x + i * config["dx(mm)"]
                y_mm = well_y + j * config["dy(mm)"]
                coordinates.append({
                    "region": well,
                    "i": i,
                    "j": j,
                    "z_level": 0,
                    "x (mm)": x_mm,
                    "y (mm)": y_mm,
                    "z (um)": 0,
                    "time": "2024-11-12_15-49-16.656512"
                })

    coordinates_df = pd.DataFrame(coordinates)
    coordinates_file = os.path.join(image_folder, "coordinates.csv")
    coordinates_df.to_csv(coordinates_file, index=False)
    return coordinates_df

def main():
    # Paths and parameters
    data_folders = [
        "/media/reef/harddisk/20250410-fucci-time-lapse-scan_2025-04-10_13-50-7.762411",
        "/media/reef/harddisk/20250410-fucci-time-lapse-scan_2025-04-10_14-50-7.948398",
        # Add other data folders here
    ]
    
    for data_folder in data_folders:
        # Extract date and time from the folder name
        folder_name = os.path.basename(data_folder)
        date_time_str = folder_name.split('_')[1] + '_' + folder_name.split('_')[2].split('.')[0]
        zarr_filename = f"{date_time_str}.zarr"
        
        image_folder = os.path.join(data_folder, "0")
        parameter_file = os.path.join(data_folder, "acquisition parameters.json")
        coordinates_file = os.path.join(image_folder, "coordinates.csv")
        output_folder = "/media/reef/harddisk/test_stitch_zarr"
        os.makedirs(output_folder, exist_ok=True)

        # Load imaging parameters and coordinates
        parameters = load_imaging_parameters(parameter_file)
        
        # Check if coordinates file exists, if not generate it
        if not os.path.exists(coordinates_file):
            print("Coordinates file not found, generating...")
            config = {
                "dx(mm)": 0.9,
                "Nx": 1,
                "dy(mm)": 0.9,
                "Ny": 1,
                "dz(um)": 1.5,
                "Nz": 1,
                "dt(s)": 0.0,
                "Nt": 1,
                "with AF": True,
                "with reflection AF": False,
                "objective": {"magnification": 20, "NA": 0.4, "tube_lens_f_mm": 180, "name": "20x"},
                "sensor_pixel_size_um": 1.85,
                "tube_lens_mm": 50
            }
            class WELLPLATE_FORMAT_96:
                NUMBER_OF_SKIP = 0
                WELL_SIZE_MM = 6.21
                WELL_SPACING_MM = 9
                A1_X_MM = 14.3
                A1_Y_MM = 11.36
            
            coordinates = generate_coordinates_file(image_folder, config, WELLPLATE_FORMAT_96)
        else:
            coordinates = pd.read_csv(coordinates_file)

        # Predefined stage limits - define the boundaries of the microscope stage
        stage_limits = {
            "x_positive": 120,
            "x_negative": 0,
            "y_positive": 86,
            "y_negative": 0,
            "z_positive": 6
        }
        
        # Get pixel size and calculate canvas size
        pixel_size_xy = get_pixel_size(parameters)
        canvas_width, canvas_height = create_canvas_size(stage_limits, pixel_size_xy)

        # Calculate appropriate number of pyramid levels
        max_dimension = max(canvas_width, canvas_height)
        max_levels = int(np.floor(np.log2(max_dimension)) // 2)  # Divide by 2 since we use 4^level
        pyramid_levels = min(max_levels, 9)  # Limit to 9 levels or less
        print(f"Using {pyramid_levels} pyramid levels for {canvas_width}x{canvas_height} canvas")
        
        # Parse image filenames and get unique channels
        image_info = parse_image_filenames(image_folder)
        channels = list(set(info["channel_name"] for info in image_info))
        print(f"Found {len(image_info)} images with {len(channels)} channels")
        selected_channel = ['Fluorescence_561_nm_Ex', 'Fluorescence_488_nm_Ex', 'BF_LED_matrix_full']
        print(f"Selected channel: {selected_channel}")

        # Check if dataset exists, if so, use it
        zarr_file_path = os.path.join(output_folder, zarr_filename)
        if os.path.exists(zarr_file_path):
            print(f"Dataset {zarr_filename} exists, opening in append mode.")
            datasets = zarr.open(zarr_file_path, mode="a")
        else:
            # Create new OME-NGFF file
            root, datasets = create_ome_ngff(output_folder, canvas_width, canvas_height, selected_channel, zarr_filename=zarr_filename, pyramid_levels=pyramid_levels)
            print(f"Dataset created with channels: {selected_channel}")

        # Process images and stitch them
        process_images(image_info, coordinates, datasets, pixel_size_xy, stage_limits, selected_channel=selected_channel, pyramid_levels=pyramid_levels)

if __name__ == "__main__":
    main()