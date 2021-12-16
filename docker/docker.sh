# This docker file path
DOCKER_FILE_PATH="docker/Dockerfile"
# Your code path, will project into docker path: /project.
CODE_PATH="/home/revolution/guangtong/mmdetection3d"

IMAGE="mmdetection_v3"
TAG="latest"
CONTAINER="gt_mm3d_v5"

function create() {
  local img_ver=$IMAGE:$TAG
  echo "Will create container: $CONTAINER, from image: $img_ver"
  xhost +  && docker run --gpus all -it -v ${CODE_PATH}:/mmdetection3d -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /data0:/mmdetection3d/data -e DISPLAY=$DISPLAY --name $CONTAINER $img_ver
}

function start() {
  xhost + && docker start $CONTAINER
}

function into() {
  echo "Will into container: $CONTAINER"
  docker exec -it $CONTAINER bash
}

function build() {
  echo "Will build docker image: $IMAGE, with tag: $TAG"
  docker build -f $DOCKER_FILE_PATH -t $IMAGE:$TAG .
}

function help() {
  echo "Run command:
          bash docker/docker.sh [Option]
        Option:
          b, build: build docker image 
          c, create: create a container from the given image 
          s, start: start the given container
          i, into: into the given container"
}

function main() {
  if [[ $1 == "build" || $1 == "b" ]]; then
    build
  elif [[ $1 == "create" || $1 == "c" ]]; then
    create
  elif [[ $1 == "start" || $1 == "s" ]]; then 
    start
  elif [[ $1 == "into" || $1 == "i" ]]; then
    into
  else 
    help
  fi
}

main "$@"