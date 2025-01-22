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
from PIL import Image, PngImagePlugin

import modules.globals

TEMP_FILE = 'temp.mp4'
TEMP_DIRECTORY = 'temp'

# monkey patch ssl for mac
if platform.system().lower() == 'darwin':
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
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error:\n{e.output.decode('utf-8')}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return False


def detect_fps(target_path: str) -> float:
    command = ['ffprobe', '-v', '0', '-select_streams', 'v:0', '-show_entries', 'stream=avg_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', target_path]
    #command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', target_path]
    output = subprocess.check_output(command).decode().strip().split('/')
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

def set_input_paths(source_path: str, target_path: str):
    try:
        modules.globals.source_path = os.path.join(modules.globals.source_folder_path, source_path)
        modules.globals.target_path = os.path.join(modules.globals.target_folder_path, target_path)
    except Exception as e:
        return f"Error constructing file paths: {e}"
    return None

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

def save_metadata(input_file_path: str, source_file_path: str, target_file_path: str):
    TargetFile = "TargetFile"
    SourceFile = "SourceFile"

    if is_video(input_file_path):
        # Adding metadata to a video
        metadata_args = [
            '-metadata', f"{TargetFile}={target_file_path}",
            '-metadata', f"{SourceFile}={source_file_path}"
        ]

        temp_file = f"{input_file_path}_tmp.mp4"

        try:
            subprocess.run(
                [
                    'ffmpeg', '-i', input_file_path,
                    *metadata_args, '-c', 'copy', temp_file, '-y'
                ],
                check=True
            )
            # Replace the original file with the new file
            os.remove(input_file_path)
            os.rename(temp_file, input_file_path)
            print("Video metadata saved successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error adding metadata to video: {e}")

    else:
        # Adding metadata to an image (must be a PNG file)
        try:
            image = Image.open(input_file_path)

            # Check if the image is in PNG format
            if image.format != "PNG":
                print("Image metadata can only be added to PNG files.")
                return

            # Create metadata object
            png_metadata = PngImagePlugin.PngInfo()

            # Add metadata
            png_metadata.add_text(TargetFile, target_file_path)
            png_metadata.add_text(SourceFile, source_file_path)

            # Save the image with metadata
            output_file = f"{os.path.splitext(input_file_path)[0]}_with_metadata.png"
            image.save(output_file, "PNG", pnginfo=png_metadata)
            print(f"Metadata saved to {output_file}")
        except Exception as e:
            print(f"Error adding metadata to image: {e}")

def read_metadata(input_file_path: str):
    """
    Reads metadata from a video or PNG image file.

    Args:
        input_file_path (str): Path to the input file.

    Returns:
        dict: A dictionary containing metadata key-value pairs, or None if not supported.
    """
    if is_video(input_file_path):
        # Reading metadata from a video
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format_tags', '-of', 'json', input_file_path],
                check=True, stdout=subprocess.PIPE, text=True
            )
            metadata = eval(result.stdout).get('format', {}).get('tags', {})
            print(f"Video Metadata: {metadata}")
            if correct_metadata_exists(metadata):
                return metadata
            else:
                return None
        except subprocess.CalledProcessError as e:
            print(f"Error reading video metadata: {e}")
            return None
    else:
        # Reading metadata from an image (must be a PNG file)
        try:
            image = Image.open(input_file_path)

            if image.format != "PNG":
                print("Image metadata can only be read from PNG files.")
                return None

            metadata = image.info
            if correct_metadata_exists(metadata):
                print(f"Image Metadata: {metadata}")
                return metadata
            else:
                return None
        except Exception as e:
            print(f"Error reading image metadata: {e}")
            return None

def correct_metadata_exists(metadata: dict) -> bool:
    """
    Checks if the metadata contains the keys TargetFile and SourceFile.

    Args:
        metadata (dict): Metadata dictionary to check.

    Returns:
        bool: True if both keys exist, False otherwise.
    """
    has_target_file = "TargetFile" in metadata
    has_source_file = "SourceFile" in metadata

    if has_target_file and has_source_file:
        print("Metadata contains TargetFile and SourceFile.")
        return True
    else:
        print("Metadata does not contain TargetFile and/or SourceFile.")
        return False
