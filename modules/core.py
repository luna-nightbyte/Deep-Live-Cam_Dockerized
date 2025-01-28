import os
import sys

import modules.client
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
import tensorflow
from concurrent.futures import ThreadPoolExecutor
from typing import List
from os import walk
from threading import Thread
import time

import modules.globals
import modules.metadata
import modules.client as client
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path, set_input_paths, save_metadata, read_metadata

if 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument('-l', '--lang', help='select output directory', dest='lang')
    program.add_argument('-server', '--server-only', help='run as server recieving and sending swapped files', dest='server_only', action='store_true', default=False)
    program.add_argument('-sf', '--source-folder', help='select a source containing source faces', dest='source_folder_path')
    program.add_argument('-tf', '--target-folder', help='select a folder containing targets', dest='target_folder_path')
    program.add_argument('-of', '--output-folder', help='select output directory', dest='output_folder_path')
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor', default=['face_swapper'], choices=['face_swapper', 'face_enhancer'], nargs='+')
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--nsfw-filter', help='filter the NSFW image or video', dest='nsfw_filter', action='store_true', default=False)
    program.add_argument('--map-faces', help='map source target faces', dest='map_faces', action='store_true', default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('--mouth-mask', help='mask the mouth region', dest='mouth_mask', action='store_true', default=False)
    program.add_argument('--live-mirror', help='The live camera display as you see it in the front-facing camera frame', dest='live_mirror', action='store_true', default=False)
    program.add_argument('--live-resizable', help='The live camera frame is resizable', dest='live_resizable', action='store_true', default=False)
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version}')

    # register deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    modules.globals.server_only = args.server_only
    if modules.globals.server_only:
        modules.globals.source_folder_path = client.DEFAULT_SOURCE_FOLDER
        modules.globals.target_folder_path = client.DEFAULT_TARGET_FOLDER
        modules.globals.output_folder_path = client.DEFAULT_OUTPUT_FOLDER
        server_thread = Thread(target=client.start_server,args=["0.0.0.0",8050], daemon=True)
        server_thread.start()
    else:
        modules.globals.source_folder_path = args.source_folder_path
        modules.globals.target_folder_path = args.target_folder_path
        modules.globals.output_folder_path = args.output_folder_path
   
    # modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path, args.output_path)
    modules.globals.frame_processors = args.frame_processor 
    modules.globals.missing_args =  ( not args.source_folder_path and not args.target_folder_path and not args.target_folder_path)
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.nsfw_filter = args.nsfw_filter
    modules.globals.map_faces = args.map_faces
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.live_mirror = args.live_mirror
    modules.globals.live_resizable = args.live_resizable
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    modules.globals.execution_threads = args.execution_threads
    modules.globals.lang = args.lang

    # translate deprecated args
    if args.source_path_deprecated:
        print('\033[33mArgument -f and --face are deprecated. Use -s and --source instead.\033[0m')
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(args.source_path_deprecated, modules.globals.target_path, args.output_path)
    if args.cpu_cores_deprecated:
        print('\033[33mArgument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == 'apple':
        print('\033[33mArgument --gpu-vendor apple is deprecated. Use --execution-provider coreml instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['coreml'])
    if args.gpu_vendor_deprecated == 'nvidia':
        print('\033[33mArgument --gpu-vendor nvidia is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['cuda'])
    if args.gpu_vendor_deprecated == 'amd':
        print('\033[33mArgument --gpu-vendor amd is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['rocm'])
    if args.gpu_threads_deprecated:
        print('\033[33mArgument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.gpu_threads_deprecated


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)
    # limit memory usage
    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = modules.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    print(f'[{scope}] {message}')





def process_video_files(processedFiles, source_file, target_file, ext_types):
    try:
    
        update_status(f"Processing [{source_file} ---> {target_file}]")
        err = set_input_paths(source_file, target_file)
        if err:
            print(err)
            return processedFiles
        
        target_file_name = os.path.join(modules.globals.target_folder_path, target_file)

        replacement_ext = None
        for ext in ext_types:
            if target_file_name.endswith(ext):
                replacement_ext = ext
                break

        if not replacement_ext:
            print(f"Unknown extension for file: {target_file_name}")
            return processedFiles
        
        # process image to videos
        if modules.globals.nsfw_filter:
            return
        if not modules.globals.map_faces:
            update_status('Creating temp resources...')
            create_temp(modules.globals.target_path)
            update_status('Extracting frames...')
            extract_frames(modules.globals.target_path)

        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
        update_status(f'Progressing... {modules.globals.output_path}')
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            frame_processor.process_frame_list(modules.globals.source_path, temp_frame_paths)
        # handles fps
        update_status('Detecting fps...')
        fps = detect_fps(modules.globals.target_path)
        update_status(f'Creating video with {fps} fps...')
        create_video(modules.globals.target_path, fps)
        # handle audio
        update_status('Restoring audio..')
        restore_audio(modules.globals.target_path, modules.globals.output_path)
        move_temp(modules.globals.target_path, modules.globals.output_path)
        # clean and validate
        clean_temp(modules.globals.target_path)
        if is_video(modules.globals.target_path):
            update_status('Processing to video succeed!') 
        else:
            update_status('Processing to video failed!')

    except Exception as e:
        print(f"Error processing files {modules.globals.source_path} and {modules.globals.target_path}: {e}")
    return processedFiles
class Targets:
    def __init__(self):
        self.targets = []  # Initialize an empty list to store targets

    def append(self, index, source_file, target_path, output_path, is_processed: bool):
        self.targets.append((index, source_file, target_path, output_path, is_processed))

    def set_processed(self,index, t):
        # Find the target with the given index and update its is_processed status
        for i, target in enumerate(self.targets):
            if target[2] == t and target[0] == index:
                # Update the tuple with is_processed set to True
                self.targets[i] = (
                    target[0],  # index
                    target[1],  # source_file
                    target[2],  # target_path
                    target[3],  # output_path
                    True,       # is_processed
                )
                return
        raise ValueError(f"Target with index {index} not found.")

    def __iter__(self):
        return iter(self.targets)
def start() -> None:
    
 
    face_enhancer_image_targets=Targets()
    
    ext_types = [".png", ".jpg", ".gif", ".bmp", ".mkv", ".mp4"]
    sourceFiles = next(walk(modules.globals.source_folder_path), (None, None, []))[2]
    targetFiles = next(walk(modules.globals.target_folder_path), (None, None, []))[2]
    update_status(f"Processing {len(sourceFiles)} source files and {len(targetFiles)} target files")
    
    image_target_files = []
    video_target_files = []

    for indexList_1, source_file in enumerate(sourceFiles):
        source_name = os.path.splitext(os.path.basename(source_file))[0]
        for target_file in targetFiles:
            modules.globals.output_path = os.path.join(modules.globals.output_folder_path, f"{source_name}_{os.path.basename(target_file)}")
            modules.globals.target_path = os.path.join(modules.globals.target_folder_path, os.path.basename(target_file))

            if has_image_extension(target_file):
                face_enhancer_image_targets.append(indexList_1, source_file, modules.globals.target_path, modules.globals.output_path, False)
                image_target_files.append(modules.globals.target_path)
            else:
                video_target_files.append(target_file)
            
    if len(image_target_files) != 0:
        for source_index, source_file in enumerate(sourceFiles):
            
            modules.globals.source_path = None
            continueloop = False
            target_files = []
            for index, source_file, target_file, output_path, is_processed in face_enhancer_image_targets:
                if source_index == index:
                    try:
                        shutil.copy2(target_file, output_path)
                        print(f"Generated output path: {modules.globals.output_path}")
                        # modules.globals.target_path = modules.globals.output_path 
                    except Exception as e:
                        print("Error copying file:", str(e), modules.globals.target_path)
                    modules.globals.source_path = os.path.join(modules.globals.source_folder_path, source_file)
                    target_files.append(output_path)

            if len(target_files) == 0:
                print("No source and/or target image(s) found..")
                continue     
            for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
                frame_processor.process_frame_list(modules.globals.source_path, target_files)

    if len(video_target_files) == 0:
        return
    
    total_files = len(sourceFiles)*len(targetFiles)
    counter = 0
    c=0
    processedFiles = []
    
    for indexList_1, source_file in enumerate(sourceFiles):
        source_name = os.path.splitext(os.path.basename(source_file))[0]
        for indexList_2, target_file in enumerate(video_target_files):
            
            
            print(source_file, target_file)
            
            target_name, extension = os.path.splitext(os.path.basename(target_file))
            modules.globals.source_path =   os.path.join(
                modules.globals.source_folder_path,
                source_file
                )
            modules.globals.target_path =   os.path.join(
                modules.globals.target_folder_path,
                target_file
                )
            modules.globals.output_path =   os.path.join(
                modules.globals.output_folder_path,
                f"{source_name}_{target_name}.{extension}"
                ) 
            counter+=1
            print(f"\nProcessing [{counter}/{total_files}]" )
            processedFiles = process_video_files(processedFiles, source_file, target_file, ext_types)
         
            c+=1
        
    release_resources()
      
def destroy(to_quit=True) -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    if to_quit: quit()

from time import sleep
def run() -> None:
    parse_args()
    if not pre_check():
        return
    
    client.wait_for_ready_signal()

    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    
    if modules.globals.server_only:
        print("Recieved files and ready to process!")
    limit_resources()
    start()
    