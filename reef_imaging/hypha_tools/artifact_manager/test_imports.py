"""Test imports to verify everything is working correctly."""

def test_imports():
    """Test importing all classes from the package."""
    try:
        # Import from main package
        from reef_imaging.hypha_tools.artifact_manager import (
            Config,
            HyphaConnection,
            UploadRecord,
            GalleryManager,
            ArtifactUploader,
            StitchManager,
            ImageFileParser,
            ImagingParameters,
            ImageProcessor,
            StitchCanvas,
            ZarrWriter
        )
        print("✅ All imports successful!")
        
        # Test instantiation of key classes
        config = Config()
        gallery_manager = GalleryManager()
        stitch_manager = StitchManager()
        print("✅ Class instantiation successful!")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_imports() 