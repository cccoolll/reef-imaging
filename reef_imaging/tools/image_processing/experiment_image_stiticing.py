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

def rotate_image(image, angle):
    """Rotate an image by the specified angle in degrees without introducing black edges."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Use a slightly larger border to avoid black edges
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
    result = rotated_padded[border_size:border_size+height, border_size:border_size+width]
    return result

def stitch_region_a1(
    image_folder,
    coordinates_file,
    parameters,
    region_name="A1",
    output_filename="stitched_result_A1.jpg",
    rotation_angle=0,
    output_scale=0.25,
    blend_overlap=True,
    overlap_margin=50  # Add extra margin to ensure proper overlap blending
):
    """
    Stitch images for region A1 based on the coordinates file.
    Images can be rotated by a specified angle before stitching.
    Images are flipped horizontally and vertically.
    Overlapping regions are blended to avoid black areas from rotation.
    The final stitched preview is saved as a JPG.

    Args:
        image_folder (str): Path to folder containing .bmp images.
        coordinates_file (str): CSV file containing columns [i,j,k,x (mm),y (mm),region].
        parameters (dict): Dictionary containing imaging parameters for pixel size calculation.
        region_name (str): Which region to stitch (e.g. 'A1').
        output_filename (str): Output JPG file name for the stitched result (preview).
        rotation_angle (float): Angle in degrees to rotate all images before stitching.
        output_scale (float): Scale factor for the final output image (e.g., 0.25 for 25% size).
        blend_overlap (bool): Whether to blend overlapping regions (default: True).
        overlap_margin (int): Extra margin in pixels to ensure better blending in overlapping areas.
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

    # 8. Create a blank canvas (8-bit grayscale for this preview) and a weight map for blending
    stitched_canvas = np.zeros((canvas_height_px, canvas_width_px), dtype=np.float32)
    weight_map = np.zeros((canvas_height_px, canvas_width_px), dtype=np.float32)

    # 9. Loop through rows in the region and place them in the canvas
    for (i, j, k, x_mm, y_mm) in rows_info:
        file_prefix = f"{region_name}_{i}_{j}_{k}"
        # naive approach: we only stitch BF channel into the final preview
        bf_name = f"{file_prefix}_BF_LED_matrix_full.bmp"
        bf_path = os.path.join(image_folder, bf_name)
        if not os.path.exists(bf_path):
            print(f"Skipping {bf_name}, file not found.")
            continue

        # Read
        img_bf = cv2.imread(bf_path, cv2.IMREAD_ANYDEPTH)
        if img_bf is None:
            print(f"Skipping {bf_path}, could not be read.")
            continue

        # Apply rotation first, before flipping
        if rotation_angle != 0:
            img_bf = rotate_image(img_bf, rotation_angle)
            
        # Then flip horizontally and vertically
        img_bf = cv2.flip(img_bf, -1)

        # Convert to float for blending
        if img_bf.dtype != np.float32:
            # Simple normalization to 0-1 range
            min_val, max_val = np.min(img_bf), np.max(img_bf)
            img_bf = (img_bf - min_val) / (max_val - min_val + 1e-5)
            img_bf = img_bf.astype(np.float32)

        # Create a weight mask for this image (higher in the center, lower at edges)
        h, w = img_bf.shape[:2]
        y, x = np.mgrid[0:h, 0:w]
        center_y, center_x = h // 2, w // 2
        
        # Use a smoother transition function with a sharper falloff at edges
        # This helps eliminate black edges in the blending
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = min(w, h) / 2
        # Smooth falloff that's mostly 1.0 in the center and drops off sharply near edges
        weight_mask = np.clip(1.0 - (dist_from_center / max_dist)**2, 0.01, 1.0)**2
        weight_mask = weight_mask.astype(np.float32)
        
        # Add a small amount of feathering at the edges to further reduce visible seams
        feather_width = int(min(w, h) * 0.05)  # 5% feather width
        if feather_width > 0:
            # Apply feathering by creating a mask that's 0 at the edges and 1 elsewhere
            edge_mask = np.ones((h, w), dtype=np.float32)
            edge_mask[:feather_width, :] *= np.linspace(0, 1, feather_width)[:, np.newaxis]
            edge_mask[-feather_width:, :] *= np.linspace(1, 0, feather_width)[:, np.newaxis]
            edge_mask[:, :feather_width] *= np.linspace(0, 1, feather_width)[np.newaxis, :]
            edge_mask[:, -feather_width:] *= np.linspace(1, 0, feather_width)[np.newaxis, :]
            
            # Apply the edge feathering to our weight mask
            weight_mask *= edge_mask

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

        # If the sub region is valid, place it with an expanded overlap area
        if sub_h > 0 and sub_w > 0:
            if blend_overlap:
                # Add weighted image to the canvas
                stitched_canvas[y_start:y_end, x_start:x_end] += (
                    img_bf[:sub_h, :sub_w] * weight_mask[:sub_h, :sub_w]
                )
                # Add weights to the weight map
                weight_map[y_start:y_end, x_start:x_end] += weight_mask[:sub_h, :sub_w]
            else:
                # Simple overwrite approach (original behavior)
                stitched_canvas[y_start:y_end, x_start:x_end] = img_bf[:sub_h, :sub_w]
                weight_map[y_start:y_end, x_start:x_end] = 1.0

    # Normalize by weight map to get the final blended result
    # Avoid division by zero by adding a small epsilon
    weight_map = np.maximum(weight_map, 1e-10)
    stitched_canvas = stitched_canvas / weight_map
    
    # Convert back to uint8 for saving
    stitched_canvas = (stitched_canvas * 255).astype(np.uint8)

    # 10. Save the final mosaic at reduced size
    if output_scale != 1.0:
        resized_height = int(stitched_canvas.shape[0] * output_scale)
        resized_width = int(stitched_canvas.shape[1] * output_scale)
        resized_canvas = cv2.resize(stitched_canvas, (resized_width, resized_height), 
                                   interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_filename, resized_canvas)
        print(f"Stitched preview saved to {output_filename} (scaled to {output_scale*100:.0f}% of original size)")
    else:
        cv2.imwrite(output_filename, stitched_canvas)
        print(f"Stitched preview saved to {output_filename}")

def main():
    # Example usage:
    data_folder = "/home/tao/europa_disk/u2os-treatment/20250410/20250410-fucci-time-lapse-scan_2025-04-10_13-50-7.762411/0"  # user to specify
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
        'pixel_size_adjustment_factor': 0.935  # Set to 1.0 by default (no adjustment)
    }

    stitch_region_a1(
        image_folder=image_folder,
        coordinates_file=coordinates_path,
        parameters=parameters,
        region_name="A1",
        output_filename="stitched_result_A1.jpg",
        rotation_angle=0.1,
        output_scale=0.7,
        blend_overlap=True,
        overlap_margin=100  # Add extra margin for better blending
    )

if __name__ == "__main__":
    main()