"""
Custom exception hierarchy for DeepLiveCam application.
Provides specific exceptions for different failure scenarios.
"""


class DeepLiveCamError(Exception):
    """Base exception for all DeepLiveCam errors."""
    pass


class ModelError(DeepLiveCamError):
    """Base exception for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Raised when a model fails to load."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a required model file is not found."""
    pass


class ModelInferenceError(ModelError):
    """Raised when model inference fails."""
    pass


class FaceDetectionError(DeepLiveCamError):
    """Raised when face detection fails."""
    pass


class NoFaceDetectedError(FaceDetectionError):
    """Raised when no face is detected in the input."""
    pass


class MultipleFacesDetectedError(FaceDetectionError):
    """Raised when multiple faces are detected but single face expected."""
    pass


class ProcessingError(DeepLiveCamError):
    """Base exception for processing errors."""
    pass


class FrameProcessingError(ProcessingError):
    """Raised when a frame fails to process."""
    pass


class VideoProcessingError(ProcessingError):
    """Raised when video processing fails."""
    pass


class ImageProcessingError(ProcessingError):
    """Raised when image processing fails."""
    pass


class FileError(DeepLiveCamError):
    """Base exception for file-related errors."""
    pass


class FileNotFoundError(FileError):
    """Raised when a required file is not found."""
    pass


class InvalidFileFormatError(FileError):
    """Raised when a file format is not supported."""
    pass


class FileSizeTooLargeError(FileError):
    """Raised when a file exceeds size limits."""
    pass


class ConfigurationError(DeepLiveCamError):
    """Raised when configuration is invalid."""
    pass


class ValidationError(DeepLiveCamError):
    """Raised when input validation fails."""
    pass


class InvalidPathError(ValidationError):
    """Raised when a file path is invalid or unsafe."""
    pass


class InvalidDimensionsError(ValidationError):
    """Raised when image/video dimensions are invalid."""
    pass


class ResourceError(DeepLiveCamError):
    """Base exception for resource-related errors."""
    pass


class InsufficientMemoryError(ResourceError):
    """Raised when system runs out of memory."""
    pass


class InsufficientDiskSpaceError(ResourceError):
    """Raised when system runs out of disk space."""
    pass


class GPUNotAvailableError(ResourceError):
    """Raised when GPU is required but not available."""
    pass


class CameraError(DeepLiveCamError):
    """Base exception for camera-related errors."""
    pass


class CameraNotFoundError(CameraError):
    """Raised when specified camera is not found."""
    pass


class CameraAccessError(CameraError):
    """Raised when camera access is denied."""
    pass


class NetworkError(DeepLiveCamError):
    """Base exception for network-related errors."""
    pass


class DownloadError(NetworkError):
    """Raised when a download fails."""
    pass


class ChecksumMismatchError(NetworkError):
    """Raised when downloaded file checksum doesn't match."""
    pass
