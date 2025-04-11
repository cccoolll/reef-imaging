import zarr
import numpy as np
import cv2

"""
Description: Access a specific tile from an OME-NGFF file and save it as a JPEG image.
"""
def preview_tile(zarr_path, output_path, channel='Fluorescence_488_nm_Ex', tile_coords=(0, 0), tile_size=(2048, 2048)):
    """
    Access a specific tile from an OME-NGFF file and save it as a JPEG image.
    
    Args:
        zarr_path (str): Path to the zarr directory
        output_path (str): Path where the tile image will be saved
        channel (str): Channel to access
        tile_coords (tuple): Coordinates of the tile (x, y)
        tile_size (tuple): Size of the tile (width, height)
    """
    # Open the zarr dataset
    store = zarr.open(zarr_path, mode='r')
    
    # Access the specified tile
    x_start, y_start = tile_coords
    width, height = tile_size
    tile_data = store[channel]['scale3'][y_start:y_start+height, x_start:x_start+width]
    
    # Save the tile as a JPEG image
    cv2.imwrite(output_path, tile_data)
    print(f"Tile image saved to: {output_path}")

# Example usage
zarr_path = "/media/reef/harddisk/test_stitch_zarr/stitched_images.zarr"
output_path = "/media/reef/harddisk/test_stitch_zarr/tile_preview.jpg"

# Preview a specific tile
preview_tile(zarr_path, output_path, channel='Fluorescence_488_nm_Ex', tile_coords=(0, 0), tile_size=(4096, 4096))