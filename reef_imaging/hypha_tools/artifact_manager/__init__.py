"""Reef Imaging Hypha Tools for Artifact Manager.

This package provides tools for interacting with the Hypha Artifact Manager,
including:
- Gallery and dataset management
- File uploading
- Image stitching and processing
"""

from .core import Config, HyphaConnection, UploadRecord
from .gallery_manager import GalleryManager
from .uploader import ArtifactUploader
from .stitch_manager import (
    ImagingParameters,
    ImageProcessor,
    StitchCanvas,
    ZarrWriter,
    ImageFileParser,
    StitchManager
)

__all__ = [
    'Config',
    'HyphaConnection', 
    'UploadRecord',
    'GalleryManager',
    'ArtifactUploader',
    'ImagingParameters',
    'ImageProcessor',
    'StitchCanvas',
    'ZarrWriter',
    'ImageFileParser',
    'StitchManager',
] 