"""
Configuration management for DeepLiveCam application.
Provides structured configuration with validation and type safety.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import yaml
from modules.constants import *


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    directory: str = MODEL_DIRECTORY
    face_swapper_fp32: str = MODEL_FACE_SWAPPER_FP32
    face_swapper_fp16: str = MODEL_FACE_SWAPPER_FP16
    face_enhancer: str = MODEL_FACE_ENHANCER
    face_analyser: str = MODEL_FACE_ANALYSER


@dataclass
class ProcessingConfig:
    """Configuration for video/image processing."""
    max_memory: int = DEFAULT_MAX_MEMORY_GB
    execution_threads: int = DEFAULT_EXECUTION_THREADS
    video_encoder: str = "libx264"
    video_quality: int = VIDEO_DEFAULT_QUALITY
    keep_fps: bool = True
    keep_audio: bool = True
    keep_frames: bool = False
    many_faces: bool = False
    map_faces: bool = False
    nsfw_filter: bool = False
    color_correction: bool = False
    mouth_mask: bool = False
    show_mouth_mask_box: bool = False


@dataclass
class UIConfig:
    """Configuration for UI settings."""
    width: int = UI_ROOT_WIDTH
    height: int = UI_ROOT_HEIGHT
    preview_width: int = UI_PREVIEW_DEFAULT_WIDTH
    preview_height: int = UI_PREVIEW_DEFAULT_HEIGHT
    live_mirror: bool = False
    live_resizable: bool = True
    show_fps: bool = False
    language: str = "en"


@dataclass
class DetectionConfig:
    """Configuration for face detection."""
    det_size: tuple = FACE_DETECTION_SIZE
    confidence_threshold: float = FACE_CONFIDENCE_THRESHOLD
    rotation_angles_coarse: List[int] = field(default_factory=lambda: list(ROTATION_ANGLES_COARSE))
    rotation_angles_fine: List[float] = field(default_factory=lambda: list(ROTATION_ANGLES_FINE))


@dataclass
class CameraConfig:
    """Configuration for camera settings."""
    default_width: int = CAMERA_DEFAULT_WIDTH
    default_height: int = CAMERA_DEFAULT_HEIGHT
    default_fps: int = CAMERA_DEFAULT_FPS


@dataclass
class AppConfig:
    """Main application configuration."""
    # Paths
    source_path: Optional[str] = None
    target_path: Optional[str] = None
    output_path: Optional[str] = None
    source_folder: Optional[str] = None
    target_folder: Optional[str] = None

    # Frame processors
    frame_processors: List[str] = field(default_factory=lambda: ['face_swapper'])
    execution_providers: List[str] = field(default_factory=list)

    # Headless mode
    headless: bool = False
    log_level: str = "INFO"

    # Sub-configurations
    models: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)

    # Runtime state (not persisted)
    source_target_map: List[Dict] = field(default_factory=list)
    simple_map: Dict = field(default_factory=dict)
    fp_ui: Dict[str, bool] = field(default_factory=lambda: {"face_enhancer": False})

    def save_to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        # Remove runtime state
        config_dict.pop('source_target_map', None)
        config_dict.pop('simple_map', None)

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)

    def save_to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        config_dict = asdict(self)
        # Remove runtime state
        config_dict.pop('source_target_map', None)
        config_dict.pop('simple_map', None)

        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @classmethod
    def load_from_json(cls, filepath: str) -> 'AppConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def load_from_yaml(cls, filepath: str) -> 'AppConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig':
        """Create AppConfig from dictionary."""
        # Handle nested configs
        if 'models' in config_dict:
            config_dict['models'] = ModelConfig(**config_dict['models'])
        if 'processing' in config_dict:
            config_dict['processing'] = ProcessingConfig(**config_dict['processing'])
        if 'ui' in config_dict:
            config_dict['ui'] = UIConfig(**config_dict['ui'])
        if 'detection' in config_dict:
            config_dict['detection'] = DetectionConfig(**config_dict['detection'])
        if 'camera' in config_dict:
            config_dict['camera'] = CameraConfig(**config_dict['camera'])

        return cls(**config_dict)

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate video quality
        if not 0 <= self.processing.video_quality <= VIDEO_MAX_QUALITY:
            errors.append(f"video_quality must be between 0 and {VIDEO_MAX_QUALITY}")

        # Validate max memory
        if self.processing.max_memory <= 0:
            errors.append("max_memory must be positive")

        # Validate execution threads
        if self.processing.execution_threads <= 0:
            errors.append("execution_threads must be positive")

        # Validate detection size
        if not (isinstance(self.detection.det_size, tuple) and len(self.detection.det_size) == 2):
            errors.append("det_size must be a tuple of (width, height)")

        # Validate paths if provided
        if self.source_path and not Path(self.source_path).exists():
            errors.append(f"source_path does not exist: {self.source_path}")

        if self.target_path and not Path(self.target_path).exists():
            errors.append(f"target_path does not exist: {self.target_path}")

        return errors


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def set_config(config: AppConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def init_config_from_file(filepath: str) -> AppConfig:
    """Initialize configuration from file."""
    path = Path(filepath)
    if path.suffix == '.json':
        config = AppConfig.load_from_json(filepath)
    elif path.suffix in ['.yml', '.yaml']:
        config = AppConfig.load_from_yaml(filepath)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")

    set_config(config)
    return config
