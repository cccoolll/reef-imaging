
import sys
sys.path.insert(0, "/home/tao/workspace/reef-imaging/reef_imaging/hypha_tools/artifact_manager/image_processing")
import stitch_zarr

# Override the data_folders with just this folder
stitch_zarr.data_folders = ["/media/reef/harddisk/20250410-fucci-time-lapse-scan_2025-04-10_15-50-8.064710"]

# Run the stitching process
stitch_zarr.main()
