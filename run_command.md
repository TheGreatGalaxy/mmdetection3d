
# Run docker
## Display issues
If report issues of matplotlib. Can try this command after installing moudles with GUI, "TKagg" .e.c.
```shell
xhost +
```

## Create a new container with image mmdetection3d.
```shell
docker run --gpus all -it -v /Dataset/:/mmdetection3d/data -v /tmp/.X11:/tmp/.X11 -e DISPLAY=$DISPLAY --name ${container_name} mmdetection3d
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
python demo/pcd_demo.py data/nuscenes/organized/sweeps/LIDAR_TOP/ checkpoints/train_2_nuscenes/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py checkpoints/train_2_nuscenes/epoch_24.pth --show --score-thr 0.25
```

## Train pointpillars on nuScenes dataset
```shell
python tools/train.py configs/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py --work-dir checkpoints/train_2_nuscenes/
```
