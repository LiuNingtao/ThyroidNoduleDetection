_base_ = [
    '../_base_/models/faster_rcnn_r50_dc5.py',
    '../_base_/datasets/coco_vid_detection_fgfa_drchong_modified.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='SELSA',
    detector=dict(
        roi_head=dict(
            type='SelsaRoIHead',
            bbox_head=dict(
                num_classes=1,
                type='SelsaBBoxHead',
                num_shared_fcs=2,
                aggregator=dict(
                    type='SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16)))))

# dataset settings
# fold = 1
data_root = '/path/to/data/root'
data = dict(
    workers_per_gpu=4,
    train=dict(
        ann_file=r'/path/to/temp_result/test.json',
        img_prefix=data_root + 'data',
    ),
    val=dict(
        ann_file=r'/path/to/temp_result/test.json',
        img_prefix=data_root + 'data',
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride')),
    test=dict(
        ann_file=r'/path/to/temp_result/test.json',
        img_prefix=data_root + 'data',
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride')))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 12, 20])
# runtime settings
total_epochs = 30
evaluation = dict(metric=['bbox'], interval=1)
