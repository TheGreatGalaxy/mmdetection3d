voxel_size = [0.2, 0.2, 6]
range_x = 50.4
range_y = 50.4
point_cloud_range = [-range_x, -range_y, -3.0, range_x, range_y, 3.0]

model = dict(
    type='PointPillars',
    # save_onnx_path="/mmdetection3d/checkpoints/train_12_nuscenes/",
    pts_voxel_layer=dict(
        max_num_points=32,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=20000,
        with_cluster_center=True,
        with_voxel_center=True),
    pts_voxel_encoder=dict(
        type='SimpleHardVFE',
        in_channels=11,
        feat_channels=[48, 64],
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[504, 504]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='BN2d', eps=0.001, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[1, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='BN2d', eps=0.001, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-range_x, -range_y, -1.80032795, range_x, range_y, -1.80032795],
                    [-range_x, -range_y, -1.6733911,
                     range_x, range_y, -1.6733911],
                    [-range_x, -range_y, -1.61785072, range_x, range_y, -1.61785072]],
            sizes=[[1.95017717, 4.60718145, 1.72270761],
                   [0.60058911, 1.68452161, 1.27192197],
                   [0.66344886, 0.7256437, 1.75748069]],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500)))
class_names = ['car', 'bicycle', 'pedestrian']
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=5,
        use_dim=[0, 1, 2, 3, 4],
        normalize_intensity=3,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=['car', 'bicycle', 'pedestrian']),
    dict(type='PointShuffle'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['car', 'bicycle', 'pedestrian']),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    # Test yw_data.
    # dict(
    #     type='LoadPointsFromFileCustom',
    #     coord_type='LIDAR',
    #     load_dim=5,
    #     use_dim=[0, 1, 2, 3, 4],
    #     normalize_intensity=3,
    #     file_client_args=dict(backend='disk')
    # ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=5,
        use_dim=[0, 1, 2, 3, 4],
        normalize_intensity=3,
        file_client_args=dict(backend='disk')),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['car', 'bicycle', 'pedestrian'],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=5,
        use_dim=[0, 1, 2, 3, 4],
        normalize_intensity=3,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['car', 'bicycle', 'pedestrian'],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        type='NuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=['car', 'bicycle', 'pedestrian'],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        with_velocity=False,
        test_mode=False,
        box_type_3d='LiDAR'),
    val=dict(
        type='NuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=['car', 'bicycle', 'pedestrian'],
        with_velocity=False,
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type='NuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=['car', 'bicycle', 'pedestrian'],
        with_velocity=False,
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR'))
evaluation = dict(
    interval=24,
    pipeline=eval_pipeline)
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[20, 23])
momentum_config = None
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'checkpoints/train_10_nuscenes/'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 1)
