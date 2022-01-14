xhost +  && docker run --gpus all -it -v /home/revolution/guangtong/mmdetection3d:/mmdetection3d \
-v /tmp/.X11-unix:/tmp/.X11-unix -v /Dataset:/mmdetection3d/data -e DISPLAY=$DISPLAY --name gt_mm3d_v2 mmdetection_v2:latest
