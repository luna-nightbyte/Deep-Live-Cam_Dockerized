import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import urllib
from pathlib import Path
from typing import List, Any
from tqdm import tqdm

import modules.globals

TEMP_FILE = "temp.mp4"
TEMP_DIRECTORY = "temp"

# monkey patch ssl for mac
if platform.system().lower() == "darwin":
    ssl._create_default_https_context = ssl._create_unverified_context


def run_ffmpeg(args: List[str]) -> bool:
    commands = [
        "ffmpeg",
        "-hide_banner",
        "-hwaccel",
        "auto",
        "-loglevel",
        modules.globals.log_level,
    ]
    commands.extend(args)
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception:
        pass
    return False


def detect_fps(target_path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        target_path,
    ]
    output = subprocess.check_output(command).decode().strip().split("/")
    try:
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception:
        pass
    return 30.0


def extract_frames(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    run_ffmpeg(
        [
            "-i",
            target_path,
            "-pix_fmt",
            "rgb24",
            os.path.join(temp_directory_path, "%04d.png"),
        ]
    )
    return temp_directory_path


def create_video(target_path: str, fps: float = 30.0) -> None:
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    run_ffmpeg(
        [
            "-r",
            str(fps),
            "-i",
            os.path.join(temp_directory_path, "%04d.png"),
            "-c:v",
            modules.globals.video_encoder,
            "-crf",
            str(modules.globals.video_quality),
            "-preset",
            "medium",      # Add the preset
            "-c:a",
            "aac",         # Add audio codec
            "-b:a",
            "128k",        # Add audio bitrate
            "-pix_fmt", 
            "yuv420p",
            "-vf",
            "colorspace=bt709:iall=bt601-6-625:fast=1",
            "-y",
            temp_output_path,
        ]
    )


def restore_audio(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg(
        [
            "-i",
            temp_output_path,
            "-i",
            target_path,
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-y",
            output_path,
        ]
    )
    if not done:
        move_temp(target_path, output_path)


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob((os.path.join(glob.escape(temp_directory_path), "*.png")))


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Any:
    if source_path and target_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(
                output_path, source_name + "-" + target_name + target_extension
            )
    return output_path


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not modules.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(("png", "jpg", "jpeg"))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith("image/"))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith("video/"))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(
            download_directory_path, os.path.basename(url)
        )
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get("Content-Length", 0))
            with tqdm(
                total=total,
                desc="Downloading",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))  # type: ignore[attr-defined]


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


import subprocess
import json
import os
import math
import tempfile
import shutil

def get_free_disk_space():
    """
    Retrieves the free disk space available on the current system.

    Returns:
        float: The free disk space in gigabytes (GB).
    """
    try:
        # Get disk usage statistics for the current partition
        total, used, free = shutil.disk_usage("/")
        # Convert bytes to gigabytes
        free_gb = free / (1024**3)
        return free_gb
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_video_info(video_path):
    """
    Retrieves video duration and frames per second (FPS) using ffprobe.

    Args:
        video_path (str): The path to the video file.

    Returns:
        tuple: A tuple containing (duration_seconds, fps).
               Returns (None, None) if information cannot be retrieved.
    """
    try:
        # Construct the ffprobe command to get stream information in JSON format
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',  # Select only the first video stream
            '-show_entries', 'stream=duration,r_frame_rate',
            '-of', 'json',
            video_path
        ]

        # Execute the command and capture the output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        duration = None
        fps = None

        if 'streams' in data and data['streams']:
            stream = data['streams'][0]

            # Get duration
            if 'duration' in stream:
                duration = float(stream['duration'])

            # Get frame rate (r_frame_rate is in 'num/den' format)
            if 'r_frame_rate' in stream:
                num, den = map(int, stream['r_frame_rate'].split('/'))
                if den != 0:
                    fps = num / den
                else:
                    print(f"Warning: Denominator of r_frame_rate is zero for {video_path}")

        return duration, fps

    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe: {e}")
        print(f"Stderr: {e.stderr}")
        return None, None
    except FileNotFoundError:
        print("Error: ffprobe not found. Please ensure FFmpeg is installed and in your system's PATH.")
        return None, None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from ffprobe output for {video_path}. Output:\n{result.stdout}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while getting video info: {e}")
        return None, None

def extract_sample_frames(video_path, num_samples, output_dir):
    """
    Extracts a specified number of frames evenly distributed across the video.

    Args:
        video_path (str): The path to the video file.
        num_samples (int): The number of frames to extract.
        output_dir (str): The directory to save the extracted frames.

    Returns:
        list: A list of paths to the extracted frame files.
              Returns an empty list if extraction fails.
    """
    duration, fps = get_video_info(video_path)
    if duration is None or fps is None:
        print(f"Could not get video info for {video_path}. Cannot extract sample frames.")
        return []

    total_frames = math.floor(duration * fps)
    if total_frames == 0:
        print(f"Video {video_path} has no frames or zero duration.")
        return []

    # Calculate intervals to get evenly distributed frames
    # We want to pick 'num_samples' frames, so we divide the total frames
    # into 'num_samples' segments and pick one frame from each segment.
    # To ensure we don't pick the very last frame (which might be incomplete/problematic)
    # or the very first frame repeatedly, we can adjust the interval.

    # Simple approach: divide total frames by num_samples to get step size
    # and pick frames at multiples of this step.
    if num_samples > total_frames:
        num_samples = total_frames # Cannot extract more frames than available

    if num_samples == 0:
        return []

    frame_interval = max(1, total_frames // num_samples)

    # Generate timestamps for extraction
    # Start from a small offset to avoid issues with frame 0
    # and ensure we get distinct frames.
    sample_timestamps = []
    for i in range(num_samples):
        # Calculate frame number
        frame_num = i * frame_interval
        if frame_num >= total_frames:
            break # Avoid going beyond video duration

        # Convert frame number to timestamp (seconds)
        timestamp_sec = frame_num / fps
        sample_timestamps.append(timestamp_sec)

    if not sample_timestamps:
        print("No valid timestamps generated for frame extraction.")
        return []

    extracted_frame_paths = []
    for i, ts in enumerate(sample_timestamps):
        frame_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")

        # Use ffmpeg to seek to the timestamp and extract one frame
        # -ss: seek to position (input option, before -i for faster seeking)
        # -i: input file
        # -vframes 1: extract only 1 frame
        # -q:v 2: quality (2 is good for JPEG, 1 is best, 31 is worst)
        cmd = [
            'ffmpeg',
            '-ss', str(ts),
            '-i', video_path,
            '-vframes', '1',
            '-q:v', '2',
            frame_path
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            if os.path.exists(frame_path):
                extracted_frame_paths.append(frame_path)
            else:
                print(f"Warning: Frame {frame_path} was not created.")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting frame at {ts}s: {e}")
            print(f"Stderr: {e.stderr}")
            # Continue to try extracting other frames even if one fails
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please ensure FFmpeg is installed and in your system's PATH.")
            return []
        except Exception as e:
            print(f"An unexpected error occurred during frame extraction: {e}")
            return []

    return extracted_frame_paths

def calculate_average_frame_size(frame_paths):
    """
    Calculates the average size (in bytes) of a list of image files.

    Args:
        frame_paths (list): A list of paths to image files.

    Returns:
        float: The average size in bytes. Returns 0 if no frames are provided.
    """
    if not frame_paths:
        return 0

    total_size = 0
    valid_frames = 0
    for path in frame_paths:
        if os.path.exists(path):
            total_size += (os.path.getsize(path) * 10)
            valid_frames += 1
        else:
            print(f"Warning: Frame file not found: {path}")

    return total_size / valid_frames if valid_frames > 0 else 0

def estimate_total_frames_size(video_path, num_sample_frames=15):
    """
    Estimates the total disk space required to extract all frames from a video.

    Args:
        video_path (str): The path to the video file.
        num_sample_frames (int): The number of sample frames to extract for
                                 average size calculation. Defaults to 15.

    Returns:
        tuple: A tuple containing (estimated_size_mb, detailed_info_dict).
               estimated_size_mb is the estimated total size in megabytes.
               detailed_info_dict contains duration, fps, total_frames,
               avg_frame_size_bytes, and num_samples_used.
               Returns (None, None) if estimation fails.
    """
    # Create a temporary directory for sample frames
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="video_frames_temp_")


        duration, fps = get_video_info(video_path)
        if duration is None or fps is None:
            return None, None

        total_frames = math.floor(duration * fps)
        if total_frames == 0:
            print(f"Video has zero frames based on duration ({duration}s) and FPS ({fps}).")
            return 0, {
                "duration_seconds": duration,
                "fps": fps,
                "total_frames": 0,
                "avg_frame_size_bytes": 0,
                "num_samples_used": 0
            }

        # Ensure we don't try to extract more samples than available frames
        actual_num_samples = min(num_sample_frames, total_frames)
        if actual_num_samples == 0:
            print("Not enough frames to extract samples for size estimation.")
            return 0, {
                "duration_seconds": duration,
                "fps": fps,
                "total_frames": total_frames,
                "avg_frame_size_bytes": 0,
                "num_samples_used": 0
            }


        sample_frame_paths = extract_sample_frames(video_path, actual_num_samples, temp_dir)

        if not sample_frame_paths:
            print("Failed to extract any sample frames. Cannot estimate size.")
            return None, None

        avg_frame_size_bytes = calculate_average_frame_size(sample_frame_paths)
        if avg_frame_size_bytes == 0:
            print("Average frame size is zero. Cannot estimate total size.")
            return None, None

        estimated_total_size_bytes = total_frames * avg_frame_size_bytes
        estimated_total_size_mb = estimated_total_size_bytes / (1024 * 1024)

        detailed_info = {
            "duration_seconds": duration,
            "fps": fps,
            "total_frames": total_frames,
            "avg_frame_size_bytes": avg_frame_size_bytes,
            "num_samples_used": len(sample_frame_paths)
        }

        return estimated_total_size_mb, detailed_info

    except Exception as e:
        print(f"An error occurred during estimation: {e}")
        return None, None
    finally:
        # Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def run_disk_check(video_file_path: str) -> None:

    free_space = get_free_disk_space()


    # Check if the dummy video exists, if not, create a very basic one for testing
    if not os.path.exists(video_file_path):
        print(f"'{video_file_path}' not found. Creating a small dummy video for testing...")
        try:
            subprocess.run([
                'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=5:size=320x240:rate=10',
                '-pix_fmt', 'yuv420p', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28',
                video_file_path
            ], check=True, capture_output=True)
            print("Dummy video created successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to create dummy video: {e.stderr}")
            print("Please ensure FFmpeg is installed and in your PATH.")
            exit()
        except FileNotFoundError:
            print("FFmpeg not found. Please install FFmpeg to create a dummy video or run the script.")
            exit()

    estimated_size_mb, details = estimate_total_frames_size(video_file_path, num_sample_frames=15)
#    if estimated_size_mb is not None:
#
#        estimated_size_gb = estimated_size_mb/1024

#        if estimated_size_gb > free_space:
#            raise OSError(f"Insufficient disk space. Estimated size needed: {estimated_size_gb:.2f} GB, Available free space: {free_space:.2f} GB.")
#            exit()

#    else:
#        print("\nCould not complete the estimation.")

