#!/bin/bash -l

NAME=${1:-patomp_wxgpt}
IMAGE=${2:-wangchangpt:v1.0}
LOCAL_VOLUME=${3:-/ist/dgx/jab}
LOCAL_PORT=${4:11888}
INSIDE_VOLUME=${5:-/workspace}

docker run \
    --restart=always -it -d --gpus=all \
    -v ${LOCAL_VOLUME}:${INSIDE_VOLUME} \
    -p ${LOCAL_PORT}:8888 \
    --name ${NAME} \
    ${IMAGE}