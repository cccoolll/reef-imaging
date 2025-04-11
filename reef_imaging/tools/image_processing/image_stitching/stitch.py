import os
import cv2
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from tqdm import tqdm
import json
import shutil  # For directory operations

def get_pixel_size(parameters, default_pixel_size=1.85, default_tube_lens_mm=50.0, default_objective_tube_lens_mm=180.0, default_magnification=40.0):
    """Calculate pixel size based on imaging parameters."""
    try:
        tube_lens_mm = float(parameters.get('tube_lens_mm', default_tube_lens_mm))
        pixel_size_um = float(parameters.get('sensor_pixel_size_um', default_pixel_size))
        objective_tube_lens_mm = float(parameters['objective'].get('tube_lens_f_mm', default_objective_tube_lens_mm))
        magnification = float(parameters['objective'].get('magnification', default_magnification))
        # Get manual adjustment factor, default to 1.0 (no adjustment)
        adjustment_factor = float(parameters.get('pixel_size_adjustment_factor', 1.0))
    except KeyError:
        raise ValueError("Missing required parameters for pixel size calculation.")

    pixel_size_xy = pixel_size_um / (magnification / (objective_tube_lens_mm / tube_lens_mm))
    # Apply manual adjustment factor
    pixel_size_xy *= adjustment_factor
    print(f"Pixel size: {pixel_size_xy} µm (with adjustment factor: {adjustment_factor})")
    return pixel_size_xy

def stitch_region_a1(
    image_folder,
    coordinates_file,
    parameters,  # Added parameters dictionary
    region_name="A1",
    output_filename="stitched_result_A1.jpg"
):
    """
    Naively stitch images for region A1 only, based on the coordinates file.
    No registration is performed. Images are flipped horizontally and vertically.
    The final stitched preview is saved as a JPG.

    Args:
        image_folder (str): Path to folder containing .bmp images.
        coordinates_file (str): CSV file containing columns [i,j,k,x (mm),y (mm),region].
        parameters (dict): Dictionary containing imaging parameters for pixel size calculation.
        region_name (str): Which region to stitch (e.g. 'A1').
        output_filename (str): Output JPG file name for the stitched result (preview).
    """
    # 1. Read the coordinates
    df = pd.read_csv(coordinates_file)

    # 2. Filter for region A1 (or whatever region_name is)
    df_region = df[df["region"] == region_name].copy()
    if df_region.empty:
        print(f"No rows found for region '{region_name}'. Exiting.")
        return

    # 3. Calculate pixel size from parameters
    pixel_size_um = get_pixel_size(parameters)
    pixel_size_mm = pixel_size_um / 1000.0

    # 4. Sort df_region according to the grid pattern described:
    #    Upper left is (I,0), upper right (I,J), bottom left (0,0), bottom right (0,J)
    #    This means I increases as you go up, J increases as you go right
    df_region.sort_values(by=["i", "j"], ascending=[False, True], inplace=True)

    # 5. Gather all image positions (in mm), track min and max
    min_x_mm = float("inf")
    min_y_mm = float("inf")
    max_x_mm = float("-inf")
    max_y_mm = float("-inf")

    # We'll store the rows in a list to keep them in index order
    rows_info = []

    for idx, row in df_region.iterrows():
        i, j, k = int(row["i"]), int(row["j"]), int(row["k"])
        x_mm, y_mm = row["x (mm)"], row["y (mm)"]
        # Keep track of bounding box in mm
        min_x_mm = min(min_x_mm, x_mm)
        min_y_mm = min(min_y_mm, y_mm)
        max_x_mm = max(max_x_mm, x_mm)
        max_y_mm = max(max_y_mm, y_mm)
        rows_info.append((i, j, k, x_mm, y_mm))

    # 6. Compute the range in mm, then compute required canvas size in pixels
    #    We'll do a simplistic approach: one image might be w x h in pixels => we need that offset.
    #    We'll assume all .bmp for A1 have the same dimension. We'll read one sample to figure that out.
    sample_name_prefix = f"{region_name}_{rows_info[0][0]}_{rows_info[0][1]}_{rows_info[0][2]}"
    # guessing any channel, let's pick BF. If not found, pick any.
    sample_bf_path = os.path.join(image_folder, f"{sample_name_prefix}_BF_LED_matrix_full.bmp")
    if not os.path.exists(sample_bf_path):
        # fallback: pick the first BMP that matches region_name
        bmp_files = [f for f in os.listdir(image_folder) if f.startswith(sample_name_prefix) and f.endswith(".bmp")]
        if bmp_files:
            sample_bf_path = os.path.join(image_folder, bmp_files[0])
        else:
            print(f"Cannot find any BMP file for sample prefix '{sample_name_prefix}'. Exiting.")
            return

    sample_img = cv2.imread(sample_bf_path, cv2.IMREAD_ANYDEPTH)
    if sample_img is None:
        print(f"Unable to read sample image {sample_bf_path}. Exiting.")
        return

    # Flip sample to confirm dimensions after flipping
    sample_img = cv2.flip(sample_img, -1)  # flip horizontally and vertically
    img_h, img_w = sample_img.shape[:2]
    # We'll define that each row's position (x_mm, y_mm) defines the top-left corner in mm.
    # Then each image extends from x_mm..(x_mm + width_in_mm).

    # 7. Translate coordinate system so the smallest X/Y is at (0,0)
    #    Then compute total canvas size in pixels
    offset_x_mm = min_x_mm
    offset_y_mm = min_y_mm

    def mm_to_px(mm_val):
        return int(round(mm_val / pixel_size_mm))

    # The farthest corner of the farthest image in mm is max_x_mm + width_in_mm
    # We'll assume each image has the same physical size in mm:
    # width_in_mm = (img_w * pixel_size_mm)
    # height_in_mm = (img_h * pixel_size_mm)

    width_in_mm = img_w * pixel_size_mm
    height_in_mm = img_h * pixel_size_mm

    max_x_mm_corner = max_x_mm + width_in_mm - offset_x_mm
    max_y_mm_corner = max_y_mm + height_in_mm - offset_y_mm

    canvas_width_px = mm_to_px(max_x_mm_corner)
    canvas_height_px = mm_to_px(max_y_mm_corner)

    print(f"Canvas size for region {region_name}: {canvas_width_px} x {canvas_height_px} px")

    # 8. Create a blank canvas (8-bit grayscale for this preview)
    stitched_canvas = np.zeros((canvas_height_px, canvas_width_px), dtype=np.uint8)

    # 9. Loop through rows in the region and place them in the canvas
    #    We'll do it for all channels we find for each i,j,k. 
    #    For the final preview, let's just handle BF_LED_matrix_full (you can adapt for others).
    #    We'll create a single final BF mosaic for demonstration.
    for (i, j, k, x_mm, y_mm) in rows_info:
        file_prefix = f"{region_name}_{i}_{j}_{k}"
        # naive approach: we only stitch BF channel into the final preview
        bf_name = f"{file_prefix}_BF_LED_matrix_full.bmp"
        bf_path = os.path.join(image_folder, bf_name)
        if not os.path.exists(bf_path):
            print(f"Skipping {bf_name}, file not found.")
            continue

        # Read and flip
        img_bf = cv2.imread(bf_path, cv2.IMREAD_ANYDEPTH)
        if img_bf is None:
            print(f"Skipping {bf_path}, could not be read.")
            continue

        # Flip horizontally and vertically
        img_bf = cv2.flip(img_bf, -1)

        # Convert 16-bit or other depth to 8-bit if needed
        if img_bf.dtype != np.uint8:
            # Simple normalization
            min_val, max_val = np.min(img_bf), np.max(img_bf)
            img_bf = (255 * (img_bf - min_val) / (max_val - min_val + 1e-5)).astype(np.uint8)

        # Determine top-left corner in the canvas
        rel_x_mm = x_mm - offset_x_mm
        rel_y_mm = y_mm - offset_y_mm
        x_start = mm_to_px(rel_x_mm)
        y_start = mm_to_px(rel_y_mm)

        # Place on canvas
        y_end = y_start + img_bf.shape[0]
        x_end = x_start + img_bf.shape[1]

        # Bound check
        if y_end > stitched_canvas.shape[0]:
            y_end = stitched_canvas.shape[0]
        if x_end > stitched_canvas.shape[1]:
            x_end = stitched_canvas.shape[1]
        sub_h = y_end - y_start
        sub_w = x_end - x_start

        # If the sub region is valid, place it
        if sub_h > 0 and sub_w > 0:
            # Possibly combine by max, but let's just overwrite for a naive approach
            stitched_canvas[y_start:y_end, x_start:x_end] = img_bf[:sub_h, :sub_w]

    # 10. Save the final mosaic
    cv2.imwrite(output_filename, stitched_canvas)
    print(f"Stitched preview saved to {output_filename}")

def create_canvas_size(stage_limits, pixel_size_xy):
    """Calculate the canvas size in pixels based on stage limits and pixel size."""
    x_range = (stage_limits["x_positive"] - stage_limits["x_negative"]) * 1000  # Convert mm to µm
    y_range = (stage_limits["y_positive"] - stage_limits["y_negative"]) * 1000  # Convert mm to µm
    canvas_width = int(x_range / pixel_size_xy)
    canvas_height = int(y_range / pixel_size_xy)
    return canvas_width, canvas_height

def load_imaging_parameters(parameter_file):
    """Load imaging parameters from a JSON file."""
    with open(parameter_file, "r") as f:
        parameters = json.load(f)
    return parameters

def create_ome_ngff(output_folder, canvas_width, canvas_height, channels, pyramid_levels=7, chunk_size=(2048, 2048)):
    """Create an OME-NGFF (zarr) file with a fixed canvas size and pyramid levels."""
    os.makedirs(output_folder, exist_ok=True)
    zarr_path = os.path.join(output_folder, "stitched_images.zarr")
    
    # If zarr exists, remove it to start fresh
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    
    # Create zarr store with proper hierarchy
    root = zarr.open_group(zarr_path, mode='w')

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
        datasets[channel].zeros(
            "scale0",
            shape=(canvas_height, canvas_width),
            chunks=chunk_size,
            dtype=np.uint8
        )

        # Create pyramid levels
        for level in range(1, pyramid_levels):
            scale_name = f"scale{level}"
            level_shape = (
                max(1, canvas_height // (2 ** level)),
                max(1, canvas_width // (2 ** level))
            )
            level_chunks = (
                min(chunk_size[0], level_shape[0]),
                min(chunk_size[1], level_shape[1])
            )

            print(f"{scale_name}: {level_shape} (chunks: {level_chunks})")

            datasets[channel].zeros(
                scale_name,
                shape=level_shape,
                chunks=level_chunks,
                dtype=np.uint8
            )

    return root, datasets

def update_pyramid(datasets, channel, level, image, x_start, y_start):
    """Update a specific pyramid level with the given image."""
    scale = 2 ** level
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

            # Combine with the existing region by taking the brighter pixel
            existing_region = pyramid_dataset[y_scaled:end_y, x_scaled:end_x]
            if existing_region.shape == scaled_image.shape:
                combined_region = np.maximum(existing_region, scaled_image)
                pyramid_dataset[y_scaled:end_y, x_scaled:end_x] = combined_region
            else:
                pyramid_dataset[y_scaled:end_y, x_scaled:end_x] = scaled_image
        else:
            print(f"Skipping update for level {level} due to out-of-bounds coordinates")

    except Exception as e:
        print(f"Error updating pyramid level {level}: {e}")
    finally:
        # Clean up
        if 'scaled_image' in locals():
            del scaled_image

def stitch_region_to_zarr(
    image_folder,
    coordinates_file,
    parameters,
    output_folder,
    region_name="A1",
    channel_names=None,
    pyramid_levels=7,
    chunk_size=(2048, 2048),
    force_recreate=False
):
    """
    Stitch images for a specific region and save to OME-ZARR format.
    
    Args:
        image_folder (str): Path to folder containing images
        coordinates_file (str): Path to CSV file with coordinates
        parameters (dict): Imaging parameters
        output_folder (str): Where to save the OME-ZARR output
        region_name (str): Which region to process
        channel_names (list): List of channel names to process
        pyramid_levels (int): Number of pyramid levels to create
        chunk_size (tuple): Chunk size for zarr storage
        force_recreate (bool): If True, recreate zarr file if exists
    """
    # Default channel names if not specified
    if channel_names is None:
        channel_names = ["BF_LED_matrix_full", "Fluorescence_488_nm_Ex", "Fluorescence_561_nm_Ex"]
        
    # 1. Read coordinates
    df = pd.read_csv(coordinates_file)
    
    # 2. Filter for the specified region
    df_region = df[df["region"] == region_name].copy()
    if df_region.empty:
        print(f"No rows found for region '{region_name}'. Exiting.")
        return
    
    # 3. Sort by i,j coordinates (upper left is high i, low j)
    df_region.sort_values(by=["i", "j"], ascending=[False, True], inplace=True)
    
    # 4. Calculate pixel size and determine canvas dimensions
    pixel_size_um = get_pixel_size(parameters)
    pixel_size_mm = pixel_size_um / 1000.0
    
    # 5. Find the canvas bounds
    min_x_mm = df_region["x (mm)"].min()
    min_y_mm = df_region["y (mm)"].min()
    max_x_mm = df_region["x (mm)"].max()
    max_y_mm = df_region["y (mm)"].max()
    
    # 6. Get sample image to determine dimensions
    sample_row = df_region.iloc[0]
    i, j, k = int(sample_row["i"]), int(sample_row["j"]), int(sample_row["k"])
    sample_name_prefix = f"{region_name}_{i}_{j}_{k}"
    
    # Try to find a sample image
    sample_path = None
    for channel in channel_names:
        potential_path = os.path.join(image_folder, f"{sample_name_prefix}_{channel}.bmp")
        if os.path.exists(potential_path):
            sample_path = potential_path
            break
    
    if sample_path is None:
        print(f"Cannot find any image files for sample prefix '{sample_name_prefix}'. Exiting.")
        return
    
    sample_img = cv2.imread(sample_path, cv2.IMREAD_ANYDEPTH)
    if sample_img is None:
        print(f"Unable to read sample image {sample_path}. Exiting.")
        return
    
    # Flip sample to confirm dimensions after flipping
    sample_img = cv2.flip(sample_img, -1)
    img_h, img_w = sample_img.shape[:2]
    
    # 7. Calculate canvas size
    width_in_mm = img_w * pixel_size_mm
    height_in_mm = img_h * pixel_size_mm
    
    max_x_mm_corner = max_x_mm + width_in_mm
    max_y_mm_corner = max_y_mm + height_in_mm
    
    offset_x_mm = min_x_mm
    offset_y_mm = min_y_mm
    
    def mm_to_px(mm_val, reference_mm):
        return int(round((mm_val - reference_mm) / pixel_size_mm))
    
    canvas_width_px = mm_to_px(max_x_mm_corner, offset_x_mm)
    canvas_height_px = mm_to_px(max_y_mm_corner, offset_y_mm)
    
    print(f"Canvas size for region {region_name}: {canvas_width_px}x{canvas_height_px} px")
    
    # 8. Create OME-ZARR structure
    os.makedirs(output_folder, exist_ok=True)
    zarr_path = os.path.join(output_folder, "stitched_images.zarr")
    
    # Check if zarr exists and handle according to force_recreate setting
    if os.path.exists(zarr_path) and force_recreate:
        print(f"Recreating zarr file at {zarr_path}")
        shutil.rmtree(zarr_path)
        root = zarr.open_group(zarr_path, mode='w')
    elif not os.path.exists(zarr_path):
        print(f"Creating new zarr file at {zarr_path}")
        root = zarr.open_group(zarr_path, mode='w')
    else:
        print(f"Opening existing zarr file at {zarr_path}")
        root = zarr.open_group(zarr_path, mode='a')
    
    # Create datasets for each channel
    datasets = {}
    for channel in channel_names:
        # Check if this channel exists in the dataset
        channel_found = False
        for row_idx, row in df_region.iterrows():
            i, j, k = int(row["i"]), int(row["j"]), int(row["k"])
            test_file = os.path.join(image_folder, f"{region_name}_{i}_{j}_{k}_{channel}.bmp")
            if os.path.exists(test_file):
                channel_found = True
                break
        
        if not channel_found:
            print(f"Channel {channel} not found in images for region {region_name}. Skipping.")
            continue
            
        print(f"\nProcessing channel: {channel}")
        
        # Check if channel exists in root
        if channel in root and not force_recreate:
            print(f"  Channel {channel} already exists")
            channel_group = root[channel]
        else:
            if channel in root:
                # Remove existing channel if force_recreate
                del root[channel]
            print(f"  Creating channel {channel}")
            channel_group = root.create_group(channel)
        
        datasets[channel] = {}
        
        # Create arrays for each scale level
        for level in range(pyramid_levels):
            scale_name = f"scale{level}"
            
            if level == 0:
                shape = (canvas_height_px, canvas_width_px)
            else:
                shape = (
                    max(1, canvas_height_px // (2 ** level)),
                    max(1, canvas_width_px // (2 ** level))
                )
                
            # Calculate appropriate chunk size
            level_chunks = (
                min(chunk_size[0], shape[0]),
                min(chunk_size[1], shape[1])
            )
            
            # Check if scale exists
            if scale_name in channel_group and not force_recreate:
                print(f"  Using existing {scale_name}: {shape}")
                datasets[channel][scale_name] = channel_group[scale_name]
            else:
                if scale_name in channel_group:
                    # Remove existing scale if force_recreate
                    del channel_group[scale_name]
                print(f"  Creating {scale_name}: {shape} (chunks: {level_chunks})")
                # Create array with v2 chunk organization
                datasets[channel][scale_name] = channel_group.zeros(
                    scale_name,
                    shape=shape,
                    chunks=level_chunks,
                    dtype=np.uint8
                )
    
    # 9. Process and add images to the zarr
    print(f"Processing {len(df_region)} positions for region {region_name}")
    
    for row_idx, row in tqdm(df_region.iterrows(), total=len(df_region)):
        i, j, k = int(row["i"]), int(row["j"]), int(row["k"])
        x_mm, y_mm = row["x (mm)"], row["y (mm)"]
        
        # Calculate pixel position
        x_start = mm_to_px(x_mm, offset_x_mm)
        y_start = mm_to_px(y_mm, offset_y_mm)
        
        # Process each channel
        for channel in datasets.keys():
            file_path = os.path.join(image_folder, f"{region_name}_{i}_{j}_{k}_{channel}.bmp")
            
            if not os.path.exists(file_path):
                # Skip if file doesn't exist for this channel
                continue
                
            # Read and flip the image
            img = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"  Skipping {file_path}, could not be read.")
                continue
                
            # Flip horizontally and vertically
            img = cv2.flip(img, -1)
            
            # Convert to 8-bit if needed
            if img.dtype != np.uint8:
                # More robust normalization using percentiles
                img_min = np.percentile(img, 1)
                img_max = np.percentile(img, 99)
                img = np.clip(img, img_min, img_max)
                img = ((img - img_min) * 255 / (img_max - img_min)).astype(np.uint8)
            
            # Calculate placement coordinates and dimensions
            y_end = min(y_start + img.shape[0], canvas_height_px)
            x_end = min(x_start + img.shape[1], canvas_width_px)
            
            # Skip if out of bounds
            if x_start >= canvas_width_px or y_start >= canvas_height_px:
                print(f"  Image at position ({i},{j}) is outside canvas bounds. Skipping.")
                continue
                
            # Clip image if needed
            if x_end - x_start != img.shape[1] or y_end - y_start != img.shape[0]:
                img = img[:y_end-y_start, :x_end-x_start]
                
            # Place image in base resolution layer
            # Combine with existing data using maximum value
            existing_region = datasets[channel]["scale0"][y_start:y_end, x_start:x_end]
            if existing_region.shape == img.shape:
                combined_region = np.maximum(existing_region, img)
                datasets[channel]["scale0"][y_start:y_end, x_start:x_end] = combined_region
            else:
                # If shapes don't match, just write what we can
                datasets[channel]["scale0"][y_start:y_end, x_start:x_end] = img
                
            # Update pyramid levels
            for level in range(1, pyramid_levels):
                update_pyramid(datasets, channel, level, img, x_start, y_start)
    
    # Clean up any 'c' directories if they exist (workaround for older zarr versions)
    for channel in channel_names:
        if channel in root:
            for level in range(pyramid_levels):
                scale_path = os.path.join(zarr_path, channel, f"scale{level}")
                c_dir = os.path.join(scale_path, "c")
                if os.path.exists(c_dir) and os.path.isdir(c_dir):
                    print(f"  Removing 'c' directory in {scale_path}")
                    # Move contents up one level
                    for item in os.listdir(c_dir):
                        src = os.path.join(c_dir, item)
                        dst = os.path.join(scale_path, item)
                        shutil.move(src, dst)
                    # Remove empty c directory
                    shutil.rmtree(c_dir)
    
    print(f"Completed OME-ZARR stitching for region {region_name}")
    return zarr_path

def main():
    # Example usage:
    data_folder = "/media/reef/harddisk/20250410-fucci-time-lapse-scan_2025-04-10_13-50-7.762411/0"  # user to specify
    image_folder = data_folder
    coordinates_path = os.path.join(data_folder, "coordinates.csv")
    parameter_file = os.path.join(data_folder, "acquisition parameters.json")
    output_folder = "/media/reef/harddisk/zarr_output"
    zarr_output_folder = os.path.join(output_folder, "zarr_output")
    
    # Create output folder if it doesn't exist
    os.makedirs(zarr_output_folder, exist_ok=True)
    
    # Load parameters
    # Check if parameter file exists
    if os.path.exists(parameter_file):
        parameters = load_imaging_parameters(parameter_file)
    else:
        print(f"Parameter file {parameter_file} not found. Using default parameters.")
        parameters = {
            'sensor_pixel_size_um': 1.85,
            'tube_lens_mm': 50.0,
            'objective': {
                'tube_lens_f_mm': 180.0,
                'magnification': 20.0  # 20x objective as mentioned in the description
            },
            'pixel_size_adjustment_factor': 0.936  # Set adjustment factor
        }

    # Generate standard preview image (JPG)
    stitch_region_a1(
        image_folder=image_folder,
        coordinates_file=coordinates_path,
        parameters=parameters,
        region_name="A1",
        output_filename=os.path.join(zarr_output_folder, "stitched_result_A1.jpg")
    )
    
    # Get unique regions from coordinates file
    df = pd.read_csv(coordinates_path)
    regions = df["region"].unique()
    
    # Ask user which regions to process
    print(f"Found {len(regions)} regions: {', '.join(regions)}")
    regions_to_process = input("Enter regions to process (comma-separated, or 'all'): ")
    
    if regions_to_process.lower() == 'all':
        regions_to_process = regions
    else:
        regions_to_process = [r.strip() for r in regions_to_process.split(',')]
    
    # Define channels to process
    default_channels = ["BF_LED_matrix_full", "Fluorescence_488_nm_Ex", "Fluorescence_561_nm_Ex"]
    
    # Detect available channels
    channels = set()
    sample_images = [f for f in os.listdir(image_folder) if f.endswith('.bmp')][:100]  # Check first 100 images
    for sample in sample_images:
        parts = sample.split('_')
        if len(parts) >= 5:
            # Extract the channel part (everything after the 4th underscore)
            channel_part = '_'.join(parts[4:]).replace('.bmp', '')
            channels.add(channel_part)
    
    print(f"Detected channels: {', '.join(channels)}")
    channels_to_process = input(f"Enter channels to process (comma-separated, or 'all'): ")
    
    if channels_to_process.lower() == 'all':
        channels_to_process = list(channels)
    else:
        channels_to_process = [c.strip() for c in channels_to_process.split(',')]
    
    # Ask if existing zarr should be recreated
    force_recreate = input("Force recreate zarr file if it exists? (y/n): ").lower() == 'y'
    
    # Process each region and create a single OME-ZARR file
    for region in regions_to_process:
        print(f"\nProcessing region: {region}")
        zarr_path = stitch_region_to_zarr(
            image_folder=image_folder,
            coordinates_file=coordinates_path,
            parameters=parameters,
            output_folder=zarr_output_folder,
            region_name=region,
            channel_names=channels_to_process,
            pyramid_levels=4,  # Adjust based on image size
            chunk_size=(2048, 2048),
            force_recreate=force_recreate
        )
        
        if zarr_path:
            print(f"Completed OME-ZARR file for region {region}: {zarr_path}")

if __name__ == "__main__":
    main()