import os
import cv2
import numpy as np
import pandas as pd

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

def main():
    # Example usage:
    data_folder = "/media/reef/harddisk/20250410-fucci-time-lapse-scan_2025-04-10_13-50-7.762411/0"  # user to specify
    image_folder = data_folder
    coordinates_path = os.path.join(data_folder, "coordinates.csv")
    
    # Example parameters dictionary
    parameters = {
        'sensor_pixel_size_um': 1.85,
        'tube_lens_mm': 50.0,
        'objective': {
            'tube_lens_f_mm': 180.0,
            'magnification': 20.0  # 20x objective as mentioned in the description
        },
        'pixel_size_adjustment_factor': 0.936  # Set to 1.0 by default (no adjustment)
    }

    stitch_region_a1(
        image_folder=image_folder,
        coordinates_file=coordinates_path,
        parameters=parameters,
        region_name="A1",
        output_filename="stitched_result_A1.jpg"
    )

if __name__ == "__main__":
    main()