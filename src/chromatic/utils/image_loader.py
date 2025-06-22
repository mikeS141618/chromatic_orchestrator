# src/chromatic/utils/image_loader.py

import os
import logging
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image


def load_images_from_directory(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
    recursive: bool = True,
    limit: Optional[int] = None
) -> List[str]:
    """Load image paths from a directory.

    Args:
        directory: Directory to load images from
        extensions: List of file extensions to include (default: common image formats)
        recursive: Whether to search subdirectories
        limit: Maximum number of images to load

    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    image_paths = []

    # Choose glob pattern based on recursive flag
    pattern = "**/*" if recursive else "*"

    for file_path in directory.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            image_paths.append(str(file_path))

            if limit and len(image_paths) >= limit:
                break

    logging.info(f"Found {len(image_paths)} images in {directory}")
    return sorted(image_paths)


def validate_image_paths(image_paths: List[str]) -> List[str]:
    """Validate that image paths exist and are readable.

    Args:
        image_paths: List of image paths to validate

    Returns:
        List of valid image paths
    """
    valid_paths = []

    for path in image_paths:
        if os.path.exists(path):
            try:
                # Try to open the image to verify it's valid
                with Image.open(path) as img:
                    img.verify()
                valid_paths.append(path)
            except Exception as e:
                logging.warning(f"Invalid image file {path}: {e}")
        else:
            logging.warning(f"Image file does not exist: {path}")

    logging.info(f"Validated {len(valid_paths)} of {len(image_paths)} images")
    return valid_paths
