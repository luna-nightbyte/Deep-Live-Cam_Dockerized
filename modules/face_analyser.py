import os
import shutil
from typing import Any,  List, Tuple
import insightface

import cv2
import numpy as np
import modules.globals
from tqdm import tqdm
from modules.typing import Frame, DetectedFace
from modules.cluster_analysis import find_cluster_centroids, find_closest_centroid
from modules.utilities import get_temp_directory_path, create_temp, extract_frames, clean_temp, get_temp_frame_paths
from pathlib import Path

FACE_ANALYSER = None
DetectedFace = dict  # Define a type alias for detected objects.  Use a dictionary


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=modules.globals.execution_providers)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER



def rotate_image(image: Frame, angle: float) -> Frame:
    """
    Rotates the image by the given angle (in degrees) around its center.

    Args:
        image: The image to rotate.
        angle: The angle of rotation in degrees.

    Returns:
        The rotated image.
    """
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # Rotation matrix
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def get_best_direction(frame: Frame) -> Tuple[float, float]:
    """
    Analyzes the frame to determine the best direction based on object detection
    scores.  It efficiently checks a few key angles.

    Args:
        frame: The input frame (image) as a NumPy array.

    Returns:
        A tuple containing:
        - The best angle (in degrees).
        - The highest detection score found at that angle.
    """
    best_angle = 0.0
    highest_score = -1.0  # Initialize with a very low score
    angles_to_check = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, -15, -30, -45, -60, -75, -90, -105, -120, -135, -150, -165, -180]  # Check straight, and diagonals.
 
    for angle in angles_to_check:
        rotated_frame = rotate_image(frame, angle) # Rotate the image
        faces = get_face_analyser().get(rotated_frame)  # Detect objects in rotated frame

        if faces:
            # Find the object with the highest detection score in the *current* rotation
            best_object_for_angle = max(faces, key=lambda x: x["det_score"])
            score = best_object_for_angle["det_score"]
            new_angles_to_check = [angle+12.5,angle+10, angle+7.5, angle+5, angle+2.5, angle-2.5, angle-5, angle-7.5, angle-10, angle-12.5]
            
            for final_angle in new_angles_to_check:
                rotated_frame = rotate_image(rotated_frame, angle) # Rotate the image
                faces = get_face_analyser().get(rotated_frame)

                if score > highest_score:
                    highest_score = score
                    best_angle = final_angle

    return best_angle, highest_score

def get_one_object(frame: Frame):
    """
    Gets the most relevant object from the frame.  Now incorporates the
    efficient angle checking.

    Args:
        frame: The input frame.

    Returns:
        The object with the highest detection score found among the checked angles,
        or None if no objects are found.
    """
    
    if frame is None:
        return None, frame
    best_angle, highest_score = get_best_direction(frame) # Get best angle and score.
    if highest_score > -1: # Check if any object was detected
        rotated_frame = rotate_image(frame, best_angle) # Rotate to best angle
        faces = get_face_analyser().get(rotated_frame) # get objects.
        # Find the object with the highest score in the rotated frame.
        best_object = max(faces, key=lambda x: x["det_score"])
        return best_object, rotated_frame
    else:
        return None, frame # No objects found
def run_detection(frame: Frame):
    """
    Gets the most relevant object from the frame.  Now incorporates the
    efficient angle checking.

    Args:
        frame: The input frame.

    Returns:
        The object with the highest detection score found among the checked angles,
        or None if no objects are found.
    """
    
    if frame is None:
        return None, frame
    best_angle, highest_score = get_best_direction(frame) # Get best angle and score.
    if highest_score > -1: # Check if any object was detected
        rotated_frame = rotate_image(frame, best_angle) # Rotate to best angle
        faces = get_face_analyser().get(rotated_frame) # get objects.
        # Find the object with the highest score in the rotated frame.
        best_object = max(faces, key=lambda x: x["det_score"])
        return best_object, rotated_frame, highest_score
    else:
        return None, frame # No objects found
    
def get_one_face(frame: Frame) -> Any:
    face = get_face_analyser().get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None

def get_many_faces(frame: Frame) -> Any:
    try:
        return get_face_analyser().get(frame)
    except IndexError:
        return None

def has_valid_map() -> bool:
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            return True
    return False

def default_source_face() -> Any:
    for map in modules.globals.source_target_map:
        if "source" in map:
            return map['source']['face']
    return None

def simplify_maps() -> Any:
    centroids = []
    faces = []
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            centroids.append(map['target']['face'].normed_embedding)
            faces.append(map['source']['face'])

    modules.globals.simple_map = {'source_faces': faces, 'target_embeddings': centroids}
    return None

def add_blank_map() -> Any:
    try:
        max_id = -1
        if len(modules.globals.source_target_map) > 0:
            max_id = max(modules.globals.source_target_map, key=lambda x: x['id'])['id']

        modules.globals.source_target_map.append({
                'id' : max_id + 1
                })
    except ValueError:
        return None
    
def get_unique_faces_from_target_image() -> Any:
    try:
        modules.globals.source_target_map = []
        target_frame = cv2.imread(modules.globals.target_path)
        many_faces = get_many_faces(target_frame)
        i = 0

        for face in many_faces:
            x_min, y_min, x_max, y_max = face['bbox']
            modules.globals.source_target_map.append({
                'id' : i, 
                'target' : {
                            'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                            'face' : face
                            }
                })
            i = i + 1
    except ValueError:
        return None
    
    
def get_unique_faces_from_target_video() -> Any:
    try:
        modules.globals.source_target_map = []
        frame_face_embeddings = []
        face_embeddings = []
    
        print('Creating temp resources...')
        clean_temp(modules.globals.target_path)
        create_temp(modules.globals.target_path)
        print('Extracting frames...')
        extract_frames(modules.globals.target_path)

        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)

        i = 0
        for temp_frame_path in tqdm(temp_frame_paths, desc="Extracting face embeddings from frames"):
            temp_frame = cv2.imread(temp_frame_path)
            many_faces = get_many_faces(temp_frame)

            for face in many_faces:
                face_embeddings.append(face.normed_embedding)
            
            frame_face_embeddings.append({'frame': i, 'faces': many_faces, 'location': temp_frame_path})
            i += 1

        centroids = find_cluster_centroids(face_embeddings)

        for frame in frame_face_embeddings:
            for face in frame['faces']:
                closest_centroid_index, _ = find_closest_centroid(centroids, face.normed_embedding)
                face['target_centroid'] = closest_centroid_index

        for i in range(len(centroids)):
            modules.globals.source_target_map.append({
                'id' : i
            })

            temp = []
            for frame in tqdm(frame_face_embeddings, desc=f"Mapping frame embeddings to centroids-{i}"):
                temp.append({'frame': frame['frame'], 'faces': [face for face in frame['faces'] if face['target_centroid'] == i], 'location': frame['location']})

            modules.globals.source_target_map[i]['target_faces_in_frame'] = temp

        # dump_faces(centroids, frame_face_embeddings)
        default_target_face()
    except ValueError:
        return None
    

def default_target_face():
    for map in modules.globals.source_target_map:
        best_face = None
        best_frame = None
        for frame in map['target_faces_in_frame']:
            if len(frame['faces']) > 0:
                best_face = frame['faces'][0]
                best_frame = frame
                break

        for frame in map['target_faces_in_frame']:
            for face in frame['faces']:
                if face['det_score'] > best_face['det_score']:
                    best_face = face
                    best_frame = frame

        x_min, y_min, x_max, y_max = best_face['bbox']

        target_frame = cv2.imread(best_frame['location'])
        map['target'] = {
                        'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                        'face' : best_face
                        }


def dump_faces(centroids: Any, frame_face_embeddings: list):
    temp_directory_path = get_temp_directory_path(modules.globals.target_path)

    for i in range(len(centroids)):
        if os.path.exists(temp_directory_path + f"/{i}") and os.path.isdir(temp_directory_path + f"/{i}"):
            shutil.rmtree(temp_directory_path + f"/{i}")
        Path(temp_directory_path + f"/{i}").mkdir(parents=True, exist_ok=True)

        for frame in tqdm(frame_face_embeddings, desc=f"Copying faces to temp/./{i}"):
            temp_frame = cv2.imread(frame['location'])

            j = 0
            for face in frame['faces']:
                if face['target_centroid'] == i:
                    x_min, y_min, x_max, y_max = face['bbox']

                    if temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)].size > 0:
                        cv2.imwrite(temp_directory_path + f"/{i}/{frame['frame']}_{j}.png", temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)])
                j += 1