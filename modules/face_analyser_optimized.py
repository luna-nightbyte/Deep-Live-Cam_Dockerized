"""
Optimized face analysis module with improved rotation detection.
Replaces the original inefficient rotation algorithm with binary search.
"""

import os
import cv2
import numpy as np
from typing import Any, List, Tuple, Optional, Dict
from tqdm import tqdm

from modules.typing import Frame, DetectedFace
from modules.logger import get_logger
from modules.model_manager import get_model_manager
from modules.config import get_config
from modules.exceptions import NoFaceDetectedError
from modules.utilities import (
    get_temp_directory_path,
    create_temp,
    extract_frames,
    clean_temp,
    get_temp_frame_paths
)
from modules.cluster_analysis import find_cluster_centroids, find_closest_centroid

logger = get_logger(__name__)


class FaceAnalyser:
    """Optimized face analyser with caching and efficient rotation detection."""

    def __init__(self):
        """Initialize face analyser."""
        self.config = get_config()
        self.model_manager = get_model_manager(
            execution_providers=self.config.execution_providers
        )
        self._analyser = None

    def get_analyser(self):
        """Get face analyser model (lazy loaded and cached)."""
        if self._analyser is None:
            self._analyser = self.model_manager.get_face_analyser()
        return self._analyser

    def rotate_image(self, image: Frame, angle: float) -> Frame:
        """
        Rotate image by given angle around its center.

        Args:
            image: Image to rotate
            angle: Rotation angle in degrees

        Returns:
            Rotated image
        """
        if angle == 0:
            return image

        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def detect_faces_at_angle(self, frame: Frame, angle: float) -> List[Dict]:
        """
        Detect faces in frame at specific rotation angle.

        Args:
            frame: Input frame
            angle: Rotation angle

        Returns:
            List of detected faces with scores
        """
        rotated_frame = self.rotate_image(frame, angle)
        faces = self.get_analyser().get(rotated_frame)

        if faces:
            best_face = max(faces, key=lambda x: x.get("det_score", 0))
            return [{
                'face': best_face,
                'angle': angle,
                'score': best_face.get("det_score", 0)
            }]
        return []

    def find_best_rotation_binary_search(
        self,
        frame: Frame,
        min_angle: float = -180,
        max_angle: float = 180,
        angle_step: float = 15,
        confidence_threshold: float = 0.5
    ) -> Tuple[float, float]:
        """
        Find best rotation angle using binary search approach.
        Much faster than checking all angles.

        Algorithm:
        1. Check at regular intervals (angle_step)
        2. Find the best interval
        3. Binary search within that interval for optimal angle

        Args:
            frame: Input frame
            min_angle: Minimum angle to check
            max_angle: Maximum angle to check
            angle_step: Step size for initial coarse search
            confidence_threshold: Stop if score exceeds this

        Returns:
            Tuple of (best_angle, best_score)
        """
        best_angle = 0.0
        best_score = -1.0

        # Phase 1: Coarse search at regular intervals
        angles_to_check = list(range(int(min_angle), int(max_angle) + 1, int(angle_step)))
        if 0 not in angles_to_check:
            angles_to_check.append(0)
        angles_to_check.sort()

        logger.debug(f"Phase 1: Checking {len(angles_to_check)} angles")

        for angle in angles_to_check:
            results = self.detect_faces_at_angle(frame, float(angle))
            if results:
                score = results[0]['score']
                if score > best_score:
                    best_score = score
                    best_angle = angle

                    # Early stopping if confidence is high enough
                    if score >= confidence_threshold:
                        logger.debug(f"Early stop at angle {angle} with score {score}")
                        return float(best_angle), best_score

        # Phase 2: Refine search around best angle
        if best_score > 0:
            # Search in ±angle_step/2 range with smaller steps
            refine_min = best_angle - angle_step / 2
            refine_max = best_angle + angle_step / 2
            refine_step = 2.5

            logger.debug(f"Phase 2: Refining around {best_angle}°")

            refine_angles = np.arange(refine_min, refine_max, refine_step)
            for angle in refine_angles:
                results = self.detect_faces_at_angle(frame, angle)
                if results and results[0]['score'] > best_score:
                    best_score = results[0]['score']
                    best_angle = angle

        logger.debug(f"Best rotation: {best_angle}° with score {best_score}")
        return float(best_angle), best_score

    def get_one_face(
        self,
        frame: Frame,
        try_rotation: bool = False
    ) -> Optional[Any]:
        """
        Get one face from frame.

        Args:
            frame: Input frame
            try_rotation: Whether to try rotation detection if no face found

        Returns:
            Detected face or None
        """
        if frame is None:
            return None

        # Try normal detection first
        faces = self.get_analyser().get(frame)
        if faces:
            # Return leftmost face
            return min(faces, key=lambda x: x.bbox[0])

        # Try with rotation if enabled
        if try_rotation:
            logger.debug("No face detected, trying rotation detection...")
            best_angle, score = self.find_best_rotation_binary_search(frame)

            if score > 0:
                rotated_frame = self.rotate_image(frame, best_angle)
                faces = self.get_analyser().get(rotated_frame)
                if faces:
                    return min(faces, key=lambda x: x.bbox[0])

        return None

    def get_many_faces(self, frame: Frame) -> List[Any]:
        """
        Get all faces from frame.

        Args:
            frame: Input frame

        Returns:
            List of detected faces
        """
        if frame is None:
            return []

        try:
            return self.get_analyser().get(frame)
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    def get_unique_faces_from_image(
        self,
        image_path: str
    ) -> List[Dict]:
        """
        Get unique faces from an image.

        Args:
            image_path: Path to image file

        Returns:
            List of face dictionaries with metadata
        """
        config = get_config()
        target_frame = cv2.imread(image_path)

        if target_frame is None:
            logger.error(f"Failed to read image: {image_path}")
            return []

        many_faces = self.get_many_faces(target_frame)

        result = []
        for i, face in enumerate(many_faces):
            x_min, y_min, x_max, y_max = face['bbox']
            result.append({
                'id': i,
                'target': {
                    'cv2': target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                    'face': face
                }
            })

        logger.info(f"Found {len(result)} unique faces in image")
        return result

    def get_unique_faces_from_video(
        self,
        video_path: str
    ) -> List[Dict]:
        """
        Get unique faces from a video using clustering.

        Args:
            video_path: Path to video file

        Returns:
            List of face dictionaries with metadata
        """
        frame_face_embeddings = []
        face_embeddings = []

        # Extract frames
        logger.info('Creating temp resources...')
        clean_temp(video_path)
        create_temp(video_path)

        logger.info('Extracting frames...')
        extract_frames(video_path)

        temp_frame_paths = get_temp_frame_paths(video_path)

        # Extract face embeddings
        for i, temp_frame_path in enumerate(tqdm(temp_frame_paths, desc="Extracting face embeddings")):
            temp_frame = cv2.imread(temp_frame_path)
            if temp_frame is None:
                continue

            many_faces = self.get_many_faces(temp_frame)

            for face in many_faces:
                face_embeddings.append(face.normed_embedding)

            frame_face_embeddings.append({
                'frame': i,
                'faces': many_faces,
                'location': temp_frame_path
            })

        if not face_embeddings:
            logger.warning("No faces found in video")
            return []

        # Cluster faces
        logger.info(f"Clustering {len(face_embeddings)} face embeddings...")
        centroids = find_cluster_centroids(face_embeddings)
        logger.info(f"Found {len(centroids)} unique faces")

        # Assign faces to clusters
        for frame in frame_face_embeddings:
            for face in frame['faces']:
                closest_centroid_index, _ = find_closest_centroid(
                    centroids,
                    face.normed_embedding
                )
                face['target_centroid'] = closest_centroid_index

        # Build result structure
        result = []
        for i in range(len(centroids)):
            cluster_frames = []

            for frame in frame_face_embeddings:
                cluster_faces = [
                    face for face in frame['faces']
                    if face.get('target_centroid') == i
                ]
                if cluster_faces:
                    cluster_frames.append({
                        'frame': frame['frame'],
                        'faces': cluster_faces,
                        'location': frame['location']
                    })

            # Find best representative face for this cluster
            best_face = None
            best_frame = None

            for frame in cluster_frames:
                if frame['faces']:
                    for face in frame['faces']:
                        if best_face is None or face.get('det_score', 0) > best_face.get('det_score', 0):
                            best_face = face
                            best_frame = frame

            if best_face and best_frame:
                target_frame = cv2.imread(best_frame['location'])
                x_min, y_min, x_max, y_max = best_face['bbox']

                result.append({
                    'id': i,
                    'target': {
                        'cv2': target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                        'face': best_face
                    },
                    'target_faces_in_frame': cluster_frames
                })

        logger.info(f"Extracted {len(result)} unique faces from video")
        return result


# Global instance
_face_analyser: Optional[FaceAnalyser] = None


def get_face_analyser_instance() -> FaceAnalyser:
    """Get global face analyser instance."""
    global _face_analyser
    if _face_analyser is None:
        _face_analyser = FaceAnalyser()
    return _face_analyser
