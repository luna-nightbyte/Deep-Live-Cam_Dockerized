import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

souce_target_map = []
simple_map = {}

source_path = None
source_folder_path = None
source_label = None

target_path = None
target_folder_path = None
target_label = None

output_path = None
output_folder_path = None
process_source_seperate = None
frame_processors: List[str] = []
keep_fps = True
keep_audio = True
keep_frames = False
many_faces = False
map_faces = False
color_correction = False  # New global variable for color correction toggle
nsfw_filter = False
video_encoder = None
video_quality = None
live_mirror = False
live_resizable = False
max_memory = None
execution_providers: List[str] = []
execution_threads = None
headless = None
missing_args = None
log_level = "error"
camera_input_combobox = None
webcam_preview_running = False
show_fps = False
mouth_mask = False
show_mouth_mask_box = False
mask_feather_ratio = 8
mask_down_size = 0.50
mask_size = 1
