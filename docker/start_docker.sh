xhost +  && docker run --gpus all -it -v /home/revolution/guangtong/mmdetection3d:/mmdetection3d \
-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v /Dataset:/mmdetection3d/data --name gt_mm3d mmdetection3d:gt_mm3d_latest
