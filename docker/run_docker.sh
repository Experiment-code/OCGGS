#!/usr/bin/env bash
# Copyright 2019 Stanford
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is modified by the author based on the original file

IMAGE_NAME=("$1")
COMMAND="bash"

WORKSPACE="/usr/TASOMerge"

echo "WORKSPACE: ${WORKSPACE}"
echo "IMAGE NAME: ${IMAGE_NAME}"
echo "DOCKER BINARY: nvidia-docker"

#nvidia-docker run --rm --pid=host \
#    -it --net=host \
#   ${IMAGE_NAME} \
#   ${COMMAND[@]}
docker run --gpus all --rm --pid=host \
    -it --net=host \
    ${IMAGE_NAME} \
    ${COMMAND[@]}