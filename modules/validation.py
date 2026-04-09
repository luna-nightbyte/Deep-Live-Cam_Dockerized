"""
Input validation and sanitization utilities.
Provides security and data integrity checks for user inputs.
"""

import mimetypes
import hashlib
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from modules.exceptions import (
    InvalidPathError,
    InvalidFileFormatError,
    FileSizeTooLargeError,
    InvalidDimensionsError,
    ValidationError
)
from modules.constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from modules.logger import get_logger

logger = get_logger(__name__)


def validate_path(
    path: str,
    must_exist: bool = True,
    allowed_extensions: Optional[Tuple[str, ...]] = None,
    base_dir: Optional[str] = None
) -> Path:
    """
    Validate and sanitize a file path.

    Args:
        path: Path to validate
        must_exist: Whether the path must exist
        allowed_extensions: Tuple of allowed file extensions
        base_dir: If provided, ensure path is within this directory

    Returns:
        Validated Path object

    Raises:
        InvalidPathError: If path is invalid or unsafe
        InvalidFileFormatError: If extension not allowed
    """
    try:
        # Convert to Path and resolve to absolute path
        path_obj = Path(path).expanduser().resolve()

        # Check for directory traversal
        if base_dir:
            base = Path(base_dir).expanduser().resolve()
            try:
                path_obj.relative_to(base)
            except ValueError:
                raise InvalidPathError(
                    f"Path '{path}' is outside allowed directory '{base_dir}'"
                )

        # Check if path exists
        if must_exist and not path_obj.exists():
            raise InvalidPathError(f"Path does not exist: {path}")

        # Check extension if provided
        if allowed_extensions and path_obj.suffix.lower() not in allowed_extensions:
            raise InvalidFileFormatError(
                f"File extension '{path_obj.suffix}' not allowed. "
                f"Allowed: {allowed_extensions}"
            )

        return path_obj

    except (OSError, RuntimeError) as e:
        raise InvalidPathError(f"Invalid path '{path}': {str(e)}")


def validate_image_path(path: str, must_exist: bool = True) -> Path:
    """
    Validate an image file path.

    Args:
        path: Path to image file
        must_exist: Whether the file must exist

    Returns:
        Validated Path object

    Raises:
        InvalidPathError: If path is invalid
        InvalidFileFormatError: If not a valid image file
    """
    return validate_path(path, must_exist, IMAGE_EXTENSIONS)


def validate_video_path(path: str, must_exist: bool = True) -> Path:
    """
    Validate a video file path.

    Args:
        path: Path to video file
        must_exist: Whether the file must exist

    Returns:
        Validated Path object

    Raises:
        InvalidPathError: If path is invalid
        InvalidFileFormatError: If not a valid video file
    """
    return validate_path(path, must_exist, VIDEO_EXTENSIONS)


def validate_media_path(path: str, must_exist: bool = True) -> Path:
    """
    Validate a media file path (image or video).

    Args:
        path: Path to media file
        must_exist: Whether the file must exist

    Returns:
        Validated Path object

    Raises:
        InvalidPathError: If path is invalid
        InvalidFileFormatError: If not a valid media file
    """
    all_extensions = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS
    return validate_path(path, must_exist, all_extensions)


def validate_file_size(
    path: Path,
    max_size_mb: Optional[int] = None
) -> int:
    """
    Validate file size.

    Args:
        path: Path to file
        max_size_mb: Maximum allowed size in MB

    Returns:
        File size in bytes

    Raises:
        FileSizeTooLargeError: If file exceeds max size
    """
    if not path.exists():
        raise InvalidPathError(f"File does not exist: {path}")

    size_bytes = path.stat().st_size

    if max_size_mb:
        max_size_bytes = max_size_mb * 1024 * 1024
        if size_bytes > max_size_bytes:
            size_mb = size_bytes / (1024 * 1024)
            raise FileSizeTooLargeError(
                f"File size ({size_mb:.1f} MB) exceeds maximum ({max_size_mb} MB)"
            )

    return size_bytes


def validate_image_dimensions(
    image: np.ndarray,
    min_width: int = 64,
    min_height: int = 64,
    max_width: int = 10000,
    max_height: int = 10000
) -> Tuple[int, int]:
    """
    Validate image dimensions.

    Args:
        image: Image array
        min_width: Minimum allowed width
        min_height: Minimum allowed height
        max_width: Maximum allowed width
        max_height: Maximum allowed height

    Returns:
        Tuple of (height, width)

    Raises:
        InvalidDimensionsError: If dimensions are invalid
    """
    if len(image.shape) < 2:
        raise InvalidDimensionsError("Image must be at least 2D")

    height, width = image.shape[:2]

    if width < min_width or height < min_height:
        raise InvalidDimensionsError(
            f"Image too small: {width}x{height}. "
            f"Minimum: {min_width}x{min_height}"
        )

    if width > max_width or height > max_height:
        raise InvalidDimensionsError(
            f"Image too large: {width}x{height}. "
            f"Maximum: {max_width}x{max_height}"
        )

    return height, width


def validate_image_file(
    path: Path,
    max_size_mb: Optional[int] = None,
    min_width: int = 64,
    min_height: int = 64
) -> Tuple[np.ndarray, int, int]:
    """
    Validate image file completely (path, size, content).

    Args:
        path: Path to image file
        max_size_mb: Maximum file size in MB
        min_width: Minimum image width
        min_height: Minimum image height

    Returns:
        Tuple of (image_array, height, width)

    Raises:
        ValidationError: If any validation fails
    """
    # Validate file size
    if max_size_mb:
        validate_file_size(path, max_size_mb)

    # Try to load image
    image = cv2.imread(str(path))
    if image is None:
        raise InvalidFileFormatError(f"Failed to load image: {path}")

    # Validate dimensions
    height, width = validate_image_dimensions(
        image, min_width, min_height
    )

    logger.debug(f"Validated image: {path} ({width}x{height})")
    return image, height, width


def validate_video_file(
    path: Path,
    max_size_mb: Optional[int] = None,
    max_duration_sec: Optional[int] = None
) -> Tuple[int, float, int, int]:
    """
    Validate video file.

    Args:
        path: Path to video file
        max_size_mb: Maximum file size in MB
        max_duration_sec: Maximum duration in seconds

    Returns:
        Tuple of (frame_count, fps, width, height)

    Raises:
        ValidationError: If any validation fails
    """
    # Validate file size
    if max_size_mb:
        validate_file_size(path, max_size_mb)

    # Try to open video
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InvalidFileFormatError(f"Failed to open video: {path}")

    try:
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Validate duration
        if max_duration_sec and fps > 0:
            duration = frame_count / fps
            if duration > max_duration_sec:
                raise ValidationError(
                    f"Video duration ({duration:.1f}s) exceeds "
                    f"maximum ({max_duration_sec}s)"
                )

        logger.debug(
            f"Validated video: {path} "
            f"({width}x{height}, {frame_count} frames @ {fps:.2f} fps)"
        )

        return frame_count, fps, width, height

    finally:
        cap.release()


def compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """
    Compute hash of a file.

    Args:
        path: Path to file
        algorithm: Hash algorithm (sha256, md5, etc.)

    Returns:
        Hex digest of file hash
    """
    hasher = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file_hash(
    path: Path,
    expected_hash: str,
    algorithm: str = "sha256"
) -> bool:
    """
    Verify file hash matches expected value.

    Args:
        path: Path to file
        expected_hash: Expected hash value
        algorithm: Hash algorithm

    Returns:
        True if hash matches

    Raises:
        ValidationError: If hash doesn't match
    """
    actual_hash = compute_file_hash(path, algorithm)
    if actual_hash.lower() != expected_hash.lower():
        raise ValidationError(
            f"File hash mismatch for {path}. "
            f"Expected: {expected_hash}, Got: {actual_hash}"
        )
    return True


def is_image_file(path: Path) -> bool:
    """Check if path is a valid image file."""
    if not path.is_file():
        return False
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: Path) -> bool:
    """Check if path is a valid video file."""
    if not path.is_file():
        return False
    return path.suffix.lower() in VIDEO_EXTENSIONS


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to remove dangerous characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove path separators
    filename = filename.replace('/', '_').replace('\\', '_')
    # Remove dangerous characters
    dangerous_chars = '<>:"|?*\0'
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    return filename or 'unnamed'
