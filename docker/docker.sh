IMAGE="mmdetection_v3"
TAG="latest"
CONTAINER="gt_mm3d_v3"

function create() {
  local img_ver=$IMAGE:$TAG
  echo "Will create container: $CONTAINER, from image: $img_ver"
  xhost +  && docker run --gpus all -it -v /home/revolution/guangtong/mmdetection3d:/mmdetection3d -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /Dataset:/mmdetection3d/data -e DISPLAY=$DISPLAY --name $CONTAINER $img_ver
}

function start() {
  docker start $CONTAINER
}


function into() {
  echo "Will into container: $CONTAINER"
  docker exec -it $CONTAINER bash
}

function help() {
  echo "bash docker/docker.sh [option]
        option: 
          c, create: create a container from the given image 
          s, start: start the given container
          i, into: into the given container"
}

function main() {
  if [[ $1 == "create" || $1 == "c" ]]; then
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