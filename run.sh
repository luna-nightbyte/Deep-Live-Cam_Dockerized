#!/bin/bash
# pip uninstall -y onnxruntime-gpu
# pip install onnxruntime onnxruntime-gpu==1.18.0 insightface
### Output paths
root_output="output"
source_dir="${root_output}/source_files" 
target_dir="${root_output}/target_files"
output_dir="${root_output}/output_files"
# Function to process a single file
process_file() {
    local source_file="$1"
    local target_file="$2"
    local output_file="${output_dir}"

    local cmd="python3 run.py \
        -sf \"${source_file}\" \
        -tf \"${target_file}\" \
        -of \"${output_file}\" \
        --max-memory ${max_mem} \
        --many-faces \
        --keep-fps \
        --frame-processor ${frame_processor} \
        --execution-threads ${threads} \
        --max-memory ${max_mem}" \

    if [[ "${USE_VIDEO_ARGS}" == true ]]; then
        cmd+=" --video-encoder ${video_encoder} --video-quality ${video_quality} --keep-fps --keep-audio"
    fi

    if [[ "${MANY_FACES}" == true ]]; then
        cmd+=" --many-faces"
    fi
    if [[ "${USE_GPU}" != false ]]; then
        cmd+=" --execution-provider ${USE_GPU}"
    fi
    if [[ "${server_only}" != false ]]; then
        cmd+=" -server"
    fi

    if [[ "${DEBUG}" == true ]]; then
        echo "Running command: $cmd"
    fi

    eval $cmd

    if [[ $? -ne 0 ]]; then
        echo "Error: Command failed for file: $source_file"
        exit 1
    fi
}

# Startup

## Create workdir folders
mkdir -p "${output_dir}"

if [[ "${DEBUG}" == true ]]; then
    debug_info
fi
process_file "$source_dir" "$target_dir"

