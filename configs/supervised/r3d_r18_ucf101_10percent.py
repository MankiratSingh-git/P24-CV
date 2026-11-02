# MMAction2 Configuration: Supervised Baseline (10% Labeled UCF-101)

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3d',
        depth=18,
        in_channels=3,
        base_channels=64,
        spatial_strides=(1, 2, 2, 2),
        inflate=(0, 0, 1, 1),
    ),
    cls_head=dict(
        type='I3DHead',
        num_classes=101,
        in_channels=512,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01
    ),
    train_cfg=dict(),
    test_cfg=dict(average_clips='score')
)

# Dataset settings
data_root = 'data/ucf101'
splits_dir = 'data/splits'
labeled_percent = 10
split = 1

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375]
)

train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1, test_mode=False),
    dict(type='RawFrameDecode'),
    dict(type='RandomResizedCrop', min_size=256, max_size=320, crop_size=224, flip_ratio=0.5, is_train=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='ToTensor', keys=['imgs', 'label']),
    dict(type='Collect', keys=['imgs', 'label'])
]

val_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='CenterCrop', crop_size=224, scale=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='ToTensor', keys=['imgs', 'label']),
    dict(type='Collect', keys=['imgs', 'label'])
]

test_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=10, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='ThreeCrop', crop_size=224, scale=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='ToTensor', keys=['imgs', 'label']),
    dict(type='Collect', keys=['imgs', 'label'])
]

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RawframeDataset',
        ann_file=f'{splits_dir}/ucf101_train_{labeled_percent}_percent_labeled_split_{split}_rawframes.txt',
        data_prefix=data_root,
        pipeline=train_pipeline
    ),
    val=dict(
        type='RawframeDataset',
        ann_file=f'{splits_dir}/ucf101_val_split_{split}_rawframes.txt',
        data_prefix=data_root,
        pipeline=val_pipeline,
        test_mode=True
    ),
    test=dict(
        type='RawframeDataset',
        ann_file=f'{splits_dir}/ucf101_val_split_{split}_rawframes.txt',
        data_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True
    )
)

# Optimizer & Scheduling
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=1e-4)
total_epochs = 120
warmup_epochs = 10
accum_steps = 2
eval_freq = 5
work_dir = 'checkpoints/supervised_baseline'
