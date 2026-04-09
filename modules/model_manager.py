"""
Model management system for DeepLiveCam.
Handles model loading, caching, and lifecycle management.
"""

from typing import Dict, Any, Optional, Callable
from pathlib import Path
import threading
import insightface
import gfpgan
import torch

from modules.logger import get_logger
from modules.exceptions import ModelLoadError, ModelNotFoundError
from modules.constants import (
    MODEL_DIRECTORY,
    MODEL_FACE_SWAPPER_FP32,
    MODEL_FACE_SWAPPER_FP16,
    MODEL_FACE_ENHANCER,
    MODEL_FACE_ANALYSER,
    FACE_DETECTION_SIZE,
    FACE_DETECTION_CTX_ID
)

logger = get_logger(__name__)


class ModelCache:
    """
    Thread-safe model cache for efficient model management.
    Implements singleton pattern to ensure single instance across application.
    """

    _instance: Optional['ModelCache'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._cache: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._initialized = True
        logger.info("Model cache initialized")

    def get(self, model_name: str) -> Optional[Any]:
        """
        Get a model from cache.

        Args:
            model_name: Name of the model

        Returns:
            Cached model or None if not cached
        """
        return self._cache.get(model_name)

    def set(self, model_name: str, model: Any) -> None:
        """
        Store a model in cache.

        Args:
            model_name: Name of the model
            model: Model instance
        """
        self._cache[model_name] = model
        logger.info(f"Model '{model_name}' cached")

    def get_or_load(
        self,
        model_name: str,
        loader_func: Callable[[], Any]
    ) -> Any:
        """
        Get model from cache or load it if not cached.

        Args:
            model_name: Name of the model
            loader_func: Function to load the model if not cached

        Returns:
            Model instance

        Raises:
            ModelLoadError: If model loading fails
        """
        # Check cache first
        model = self.get(model_name)
        if model is not None:
            logger.debug(f"Model '{model_name}' loaded from cache")
            return model

        # Get or create lock for this model
        if model_name not in self._locks:
            with self._lock:
                if model_name not in self._locks:
                    self._locks[model_name] = threading.Lock()

        # Load model with lock
        with self._locks[model_name]:
            # Double-check cache (another thread may have loaded it)
            model = self.get(model_name)
            if model is not None:
                return model

            # Load model
            logger.info(f"Loading model '{model_name}'...")
            try:
                model = loader_func()
                self.set(model_name, model)
                logger.info(f"Model '{model_name}' loaded successfully")
                return model
            except Exception as e:
                logger.error(f"Failed to load model '{model_name}': {e}", exc_info=True)
                raise ModelLoadError(f"Failed to load model '{model_name}': {str(e)}")

    def clear(self, model_name: Optional[str] = None) -> None:
        """
        Clear cached models.

        Args:
            model_name: Specific model to clear, or None to clear all
        """
        if model_name:
            self._cache.pop(model_name, None)
            logger.info(f"Cleared model '{model_name}' from cache")
        else:
            self._cache.clear()
            logger.info("Cleared all models from cache")

    def list_cached(self) -> list:
        """Get list of cached model names."""
        return list(self._cache.keys())

    def get_memory_usage(self) -> Dict[str, int]:
        """
        Estimate memory usage of cached models.

        Returns:
            Dict mapping model names to estimated memory usage in bytes
        """
        # This is a rough estimate
        usage = {}
        for name, model in self._cache.items():
            try:
                # Try to get actual size for torch models
                if hasattr(model, 'parameters'):
                    size = sum(p.numel() * p.element_size() for p in model.parameters())
                else:
                    # Rough estimate
                    size = 0
                usage[name] = size
            except Exception:
                usage[name] = 0
        return usage


class ModelManager:
    """High-level interface for model management."""

    def __init__(
        self,
        model_dir: str = MODEL_DIRECTORY,
        execution_providers: Optional[list] = None
    ):
        """
        Initialize model manager.

        Args:
            model_dir: Directory containing model files
            execution_providers: List of execution providers for ONNX models
        """
        self.model_dir = Path(model_dir)
        self.execution_providers = execution_providers or ['CPUExecutionProvider']
        self.cache = ModelCache()

        logger.info(f"Model manager initialized with dir: {model_dir}")
        logger.info(f"Execution providers: {self.execution_providers}")

    def get_face_analyser(self) -> Any:
        """
        Get face analysis model.

        Returns:
            FaceAnalysis model instance
        """
        def loader():
            logger.info(f"Loading face analyser '{MODEL_FACE_ANALYSER}'...")
            analyser = insightface.app.FaceAnalysis(
                name=MODEL_FACE_ANALYSER,
                providers=self.execution_providers
            )
            analyser.prepare(ctx_id=FACE_DETECTION_CTX_ID, det_size=FACE_DETECTION_SIZE)
            return analyser

        return self.cache.get_or_load('face_analyser', loader)

    def get_face_swapper(self) -> Any:
        """
        Get face swapper model.
        Prioritizes FP32 model over FP16 for better quality.

        Returns:
            Face swapper model instance

        Raises:
            ModelNotFoundError: If neither FP32 nor FP16 model found
            ModelLoadError: If model loading fails
        """
        def loader():
            # Check for FP32 model first
            model_path_fp32 = self.model_dir / MODEL_FACE_SWAPPER_FP32
            model_path_fp16 = self.model_dir / MODEL_FACE_SWAPPER_FP16

            chosen_path = None
            if model_path_fp32.exists():
                chosen_path = model_path_fp32
                logger.info(f"Using FP32 face swapper model: {model_path_fp32}")
            elif model_path_fp16.exists():
                chosen_path = model_path_fp16
                logger.info(f"Using FP16 face swapper model: {model_path_fp16}")
            else:
                raise ModelNotFoundError(
                    f"Face swapper model not found. Checked: {model_path_fp32}, {model_path_fp16}"
                )

            swapper = insightface.model_zoo.get_model(
                str(chosen_path),
                providers=self.execution_providers
            )
            return swapper

        return self.cache.get_or_load('face_swapper', loader)

    def get_face_enhancer(self) -> Any:
        """
        Get face enhancement model.

        Returns:
            GFPGAN model instance

        Raises:
            ModelNotFoundError: If model not found
            ModelLoadError: If model loading fails
        """
        def loader():
            model_path = self.model_dir / MODEL_FACE_ENHANCER
            if not model_path.exists():
                raise ModelNotFoundError(
                    f"Face enhancer model not found: {model_path}"
                )

            logger.info(f"Loading face enhancer: {model_path}")

            # Determine device
            device = 'cpu'
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'

            logger.info(f"Using device for face enhancer: {device}")

            if device == 'mps':
                mps_device = torch.device("mps")
                enhancer = gfpgan.GFPGANer(
                    model_path=str(model_path),
                    upscale=1,
                    device=mps_device
                )
            else:
                enhancer = gfpgan.GFPGANer(
                    model_path=str(model_path),
                    upscale=1
                )

            return enhancer

        return self.cache.get_or_load('face_enhancer', loader)

    def preload_models(self, model_names: Optional[list] = None) -> None:
        """
        Preload models into cache.

        Args:
            model_names: List of model names to preload, or None to preload all
        """
        if model_names is None:
            model_names = ['face_analyser', 'face_swapper', 'face_enhancer']

        logger.info(f"Preloading models: {model_names}")

        for model_name in model_names:
            try:
                if model_name == 'face_analyser':
                    self.get_face_analyser()
                elif model_name == 'face_swapper':
                    self.get_face_swapper()
                elif model_name == 'face_enhancer':
                    self.get_face_enhancer()
                else:
                    logger.warning(f"Unknown model name: {model_name}")
            except Exception as e:
                logger.error(f"Failed to preload {model_name}: {e}")

    def clear_cache(self) -> None:
        """Clear all cached models."""
        self.cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached models.

        Returns:
            Dict with cache information
        """
        cached_models = self.cache.list_cached()
        memory_usage = self.cache.get_memory_usage()

        return {
            'cached_models': cached_models,
            'model_count': len(cached_models),
            'memory_usage': memory_usage,
            'total_memory_mb': sum(memory_usage.values()) / (1024 * 1024)
        }


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager(
    model_dir: str = MODEL_DIRECTORY,
    execution_providers: Optional[list] = None
) -> ModelManager:
    """
    Get global model manager instance.

    Args:
        model_dir: Directory containing models
        execution_providers: Execution providers for ONNX

    Returns:
        ModelManager instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(model_dir, execution_providers)
    return _model_manager
