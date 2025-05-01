import zarr
import numpy as np
import cv2

"""
Description: Access a specific chunk from an OME-NGFF file and save it as a JPEG image.
"""
def preview_chunk(zarr_path, output_path, channel='Fluorescence_405_nm_Ex', 
                 chunk_coords=(0, 0), chunk_size=(256, 256), scale=3,
                 use_histogram_eq=False, clip_percentiles=(0.5, 99.5)):
    """
    Access a specific chunk from an OME-NGFF file and save it as a JPEG image.
    
    Args:
        zarr_path (str): Path to the zarr directory
        output_path (str): Path where the chunk image will be saved
        channel (str): Channel to access
        chunk_coords (tuple): Coordinates of the chunk (x, y)
        chunk_size (tuple): Size of the chunk (width, height)
        scale (int): Scale level to use (0 is full resolution, higher values are downsampled)
        use_histogram_eq (bool): Whether to apply histogram equalization (can cause overexposure)
        clip_percentiles (tuple): Percentiles for contrast stretching (min, max)
    """
    # Open the zarr dataset
    store = zarr.open(zarr_path, mode='r')
    
    # Access the specified chunk
    x_start, y_start = chunk_coords
    width, height = chunk_size
    
    scale_name = f'scale{scale}'
    print(f"Accessing channel '{channel}' at {scale_name}, position ({x_start}, {y_start}), size {width}x{height}")
    
    # Check if channel exists
    if channel not in store:
        print(f"Channel '{channel}' not found in zarr. Available channels: {list(store.keys())}")
        return
    
    # Check if scale exists
    if scale_name not in store[channel]:
        print(f"Scale '{scale_name}' not found. Available scales: {list(store[channel].keys())}")
        return
    
    # Check data shape and adjust if needed
    data_shape = store[channel][scale_name].shape
    print(f"Full dataset shape: {data_shape}")
    
    # Make sure we don't try to read beyond the data bounds
    y_end = min(y_start + height, data_shape[0])
    x_end = min(x_start + width, data_shape[1])
    
    if y_end <= y_start or x_end <= x_start:
        print("Requested chunk is outside dataset bounds")
        return
    
    # Read the data
    chunk_data = store[channel][scale_name][y_start:y_end, x_start:x_end]
    print(f"Read chunk shape: {chunk_data.shape}, dtype: {chunk_data.dtype}, min: {chunk_data.min()}, max: {chunk_data.max()}")
    
    # Normalize the image to improve visibility if it's very dark
    if chunk_data.max() > 0:  # Only normalize if there's actual data
        # More conservative percentile normalization
        p_low, p_high = np.percentile(chunk_data, clip_percentiles)
        if p_high > p_low:
            # Scale to 0-255 range
            normalized = np.clip((chunk_data - p_low) * 255.0 / (p_high - p_low), 0, 255).astype(np.uint8)
        else:
            normalized = chunk_data.astype(np.uint8)
    else:
        print("Warning: Image appears to be empty (all zeros)")
        normalized = chunk_data.astype(np.uint8)
    
    # Apply histogram equalization only if requested
    if use_histogram_eq:
        output_img = cv2.equalizeHist(normalized)
    else:
        # Optional: Apply a milder contrast enhancement using CLAHE
        # (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        output_img = clahe.apply(normalized)
    
    # Save the chunk as a JPEG image
    cv2.imwrite(output_path, output_img)
    print(f"Chunk image saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    zarr_path = "/media/reef/harddisk/test_stitch_zarr/2025-04-29_15-38-36.zarr"
    output_path = "/media/reef/harddisk/test_stitch_zarr/chunk_preview.jpg"

    # Preview a specific chunk - try different scales and positions if needed
    preview_chunk(zarr_path, output_path, 
                 channel='BF_LED_matrix_full', 
                 chunk_coords=(0, 0), 
                 chunk_size=(6015, 6015),
                 scale=3,                      # Try scale 0, 1, 2, etc.
                 use_histogram_eq=False,       # Set to False to avoid overexposure
                 clip_percentiles=(0.5, 99.5)) # More conservative clipping