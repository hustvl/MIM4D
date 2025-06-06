_base_ = [
    "../../../configs/_base_/datasets/nus-3d.py",
    "../../../configs/_base_/default_runtime.py",
]

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
work_dir = './work_dir/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
unified_voxel_size = [0.8, 0.8, 1.6]
frustum_range = [-1, -1, 0.0, -1, -1, 64.0]
frustum_size = [-1, -1, 1.0]
cam_sweep_num = 3
HOP = True
fp16_enabled = True
unified_voxel_shape = [
    int((point_cloud_range[3] - point_cloud_range[0]) / unified_voxel_size[0]),
    int((point_cloud_range[4] - point_cloud_range[1]) / unified_voxel_size[1]),
    int((point_cloud_range[5] - point_cloud_range[2]) / unified_voxel_size[2]),
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    cam_sweep_num=cam_sweep_num,
)

model = dict(
    type="UVTRSSL",
    cam_sweep_num=cam_sweep_num,
    HOP = HOP,
    img_backbone=dict(
        type="MaskConvNeXt",
        arch="base",
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        norm_out=True,
        frozen_stages=1,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="ckpts/convnextB_1kpretrained_official_style.pth",
        ),
        mae_cfg=dict(
            downsample_scale=32, downsample_dim=768, mask_ratio=0.3, learnable=False
        ),
    ),
    img_neck=dict(
        type="FPN",
        in_channels=[128, 256, 512, 1024],
        out_channels=128,
        start_level=1,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    # img_backbone=dict(
    #     type='MaskResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',
    #     init_cfg=dict(type='Pretrained', checkpoint='ckpts/resnet50-19c8e357.pth'),
    #     mae_cfg=dict(downsample_scale=32, downsample_dim=768, mask_ratio=0.3, learnable=False),
    #     ),
    # img_neck=dict(
    #     type='FPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=256,
    #     start_level=0,
    #     add_extra_convs='on_output',
    #     num_outs=4,
    #     relu_before_extra_convs=True),
    depth_head=dict(type="SimpleDepth"),
    view_trans=dict(
            type="Uni3DViewTrans",
            pc_range=point_cloud_range,
            voxel_size=unified_voxel_size,
            voxel_shape=unified_voxel_shape,
            frustum_range=frustum_range,
            frustum_size=frustum_size,
            num_convs=0,
            keep_sweep_dim=False,
            fp16_enabled=fp16_enabled,
            sweep_fusion='None',
        ),
    render_conv_cfg=dict(in_channels=128, out_channels=32, kernel_size=3, padding=1),
    pts_bbox_head=dict(
        type="RenderHead",
        fp16_enabled=fp16_enabled,
        in_channels=128,
        unified_voxel_size=unified_voxel_size,
        unified_voxel_shape=unified_voxel_shape,
        pc_range=point_cloud_range,
        view_cfg = None,
        render_conv_cfg=dict(out_channels=32, kernel_size=3, padding=1),
        ray_sampler_cfg=dict(
            close_radius=3.0,
            far_radius=50.0,
            only_img_mask=False,
            only_point_mask=False,
            replace_sample=False,
            point_nsample=2048,
            point_ratio=0.5,
            pixel_interval=4,
            sky_region=0.4,
            merged_nsample=2048,
        ),
        render_ssl_cfg=dict(
            type="NeuSModel",
            norm_scene=True,
            field_cfg=dict(
                type="SDFField",
                sdf_decoder_cfg=dict(
                    in_dim=32, out_dim=16 + 1, hidden_size=16, n_blocks=5
                ),
                rgb_decoder_cfg=dict(
                    in_dim=32 + 16 + 3 + 3, out_dim=3, hidden_size=16, n_blocks=3
                ),
                interpolate_cfg=dict(type="SmoothSampler", padding_mode="zeros"),
                beta_init=0.3,
            ),
            collider_cfg=dict(type="AABBBoxCollider", near_plane=1.0),
            sampler_cfg=dict(
                type="NeuSSampler",
                initial_sampler="UniformSampler",
                num_samples=72,
                num_samples_importance=24,
                num_upsample_steps=1,
                train_stratified=True,
                single_jitter=True,
            ),
            loss_cfg=dict(
                sensor_depth_truncation=0.1,
                sparse_points_sdf_supervised=False,
                weights=dict(
                    depth_loss=10.0,
                    rgb_loss=10.0,
                ),
            ),
        ),
    ),
    history_decoder=dict(
        type='BiTemporalPredictor_longshort',
        in_channels=160, #32*11
        out_channels=160,
        embed_dims=128,
        num_adj=cam_sweep_num-1,
        reduction=2,
        bev_h=128, 
        bev_w=128,
        decoder_short=dict(
            type='TemporalDecoder',
            num_layers=2,
            transformerlayers=dict(
                type='BEVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TemporalCrossAttention',
                        embed_dims=128,
                        num_heads=4,
                        num_levels=1,
                        num_bev_queue=1,
                        dropout=0.0)
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=128,
                    feedforward_channels=256,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                feedforward_channels=256,
                ffn_dropout=0.0,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        decoder_long=dict(
            type='TemporalDecoder',
            num_layers=2,
            transformerlayers=dict(
                type='BEVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TemporalCrossAttention',
                        embed_dims=128//2,
                        num_heads=2,
                        # embed_dims=128,
                        # num_heads=4,
                        num_levels=1,
                        num_bev_queue=cam_sweep_num-1,
                        dropout=0.0)
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=128//2,
                    feedforward_channels=128,
                    # embed_dims=128,
                    # feedforward_channels=256,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                feedforward_channels=256,
                ffn_dropout=0.0,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                ),
)

dataset_type = "NuScenesSweepDataset"
data_root = "./data/nuscenes/"

file_client_args = dict(backend="disk")


train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=10,
    #     # use_dim=[0, 1, 2, 3, 4],
    #     # pad_empty_sweeps=True,
    #     # remove_close=True
    #     ),
    dict(
        type="LoadMultiViewMultiSweepImageFromFiles",
        sweep_num=cam_sweep_num,
        to_float32=True,
        file_client_args=file_client_args,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PointShuffle"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="CollectUnified3D", keys=["points", "img"]),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=10,
    #     # use_dim=[0, 1, 2, 3, 4],
    #     # pad_empty_sweeps=True,
    #     # remove_close=True
    #     ),
    dict(
        type="LoadMultiViewMultiSweepImageFromFiles",
        sweep_num=cam_sweep_num,
        to_float32=True,
        file_client_args=file_client_args,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PointShuffle"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="CollectUnified3D", keys=["points", "img"]),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root
        + "nuscenes_unified_infos_train.pkl",  # please change to your own info file
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d="LiDAR",
        load_interval=1,
    ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + "nuscenes_unified_infos_val.pkl",
        load_interval=1,
    ),  # please change to your own info file
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + "nuscenes_unified_infos_val.pkl",
        load_interval=1,
    ),
)  # please change to your own info file

optimizer = dict(
    type="AdamW",
    lr=4e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 12
evaluation = dict(interval=50, pipeline=test_pipeline)
checkpoint_config = dict(max_keep_ckpts=1, interval=1)
log_config = dict(
    interval=100, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

find_unused_parameters = True
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
custom_hooks = [
    dict(
        type='EMAHook',),
]
# fp16 setting
fp16 = dict(loss_scale=32.0)
