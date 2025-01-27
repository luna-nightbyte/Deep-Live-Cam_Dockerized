#!/bin/bash

# Uncomment below if onnxruntime fails. Once sucessful, comment out again.
# pip uninstall -y onnxruntime-gpu
# pip install onnxruntime onnxruntime-gpu==1.18.0 insightface


if [ -f .env ]; then
    source .env
else
    echo ".env file not found! Copy [ example.env -> .env ] and make sure all settings are correct."
    exit 1
fi

cmd="python3 run.py \
    -sf \"${SOURCE_FOLDER}\" \
    -tf \"${TARGET_FOLDER}\" \
    -of \"${OUTPUT_FOLDER}\" \
    --max-memory ${MAX_MEM} \
    --many-faces \
    --keep-fps \
    --frame-processor {${FRAME_PROCESSOR}} \
    --execution-threads ${THREADS}"

if [[ "${USE_VIDEO_ARGS}" == true ]]; then
    cmd+=" \
    --video-encoder ${VIDEO_ENCODER} \
     --video-quality ${VIDEO_QUALITY} \
     --keep-fps --keep-audio"
fi
if [[ "${MANY_FACES}" == true ]]; then
    cmd+=" \
    --many-faces"
fi
if [[ "${USE_GPU}" != false ]]; then
    cmd+=" \
    --execution-provider ${RUNTIME}"
fi
if [[ "${SERVER_ONLY}" != false ]]; then
    cmd+=" \
    -server"
fi
eval $cmd
if [[ $? -ne 0 ]]; then
    echo "Error: Command failed for file: $source_file"
    exit 1
fi