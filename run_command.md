# Build docker
> docker build -f docker/Dockerfile -t mmdetection_v3 .

Create a container, and into it:
> bash docker/start_docker.sh

Install python dependencies:
> pip install -r requirements.txt

Install mmdetection3d:
> pip install -v -e . 

Color terminator:
> echo "export PS1='${debian_chroot:+($debian_chroot)}\[\033[01;35;01m\]\u\[\033[00;31;01m\]@\[\033[01;36;01m\]\h\[\033[00;31;01m\]:\[\033[00;00;01m\]\w \[\033[01;32;01m\]\$ \[\033[01;33;01m\]'" >> ~/.bashrc

Check train log by tensorboard
> tensorboard --logdir=checkpoints/train_23_nuscenes/tf_logs/.

If appears errors about libgl when using open3d, maybe you should install libnvidia_gl lib, run the command in docker terminal:
> apt install libnvidia-gl-xxx
'xxx' is the version of cuda driver.

# Run in docker
## Display issues
If report issues of matplotlib. Can try this command after installing moudles with GUI, "TKagg" .e.c.
```shell
xhost +
```

## Create a new container with image mmdetection3d.
```shell
docker run --gpus all -it -v ${dataset_path}:/mmdetection3d/data  -v ${mmdetection3d_code_path}:/mmdetection3d -v /tmp/.X11:/tmp/.X11 -e DISPLAY=$DISPLAY --name ${container_name} mmdetection3d
```

# Run models
## Run pointpillars on kitti dataset
Detect a single bin file:
```shell
python demo/pcd_demo.py demo/data/kitti/kitti_000008.bin configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py \
checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth
```

Detect a single bin file and show pointcloud:
```shell
python demo/pcd_demo.py demo/data/kitti/kitti_000008.bin configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py \
checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth --show
```


## Run pointpillars on nuScenes dataset
Detect multiple bin files under given directory, and show results with score threshold.
```shell
python demo/pcd_demo.py data/nuscenes/sweeps/LIDAR_TOP/ checkpoints/train_24_nuscenes/point_pillars_hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py checkpoints/train_24_nuscenes/epoch_24.pth --show --score-thr 0.25
```

## Train pointpillars on nuScenes dataset
```shell
python tools/train.py configs/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py --work-dir checkpoints/train_2_nuscenes/
```
