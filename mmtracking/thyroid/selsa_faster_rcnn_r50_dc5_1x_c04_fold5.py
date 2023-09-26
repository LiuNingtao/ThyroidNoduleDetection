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
fold = 5
data_root = '/path/to/data/root'
data = dict(
    workers_per_gpu=4,
    train=dict(
        ann_file=data_root + f'dataset/cross_valid/{str(fold)}/train.json',
        img_prefix=data_root + 'data',
        ref_img_sampler=dict(
            num_ref_imgs=8,
            frame_range=21)
    ),
    val=dict(
        ann_file=data_root + f'dataset/cross_valid/{str(fold)}/test.json',
        img_prefix=data_root + 'data',
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=20,
            frame_range=[-10,0],
            method='test_with_adaptive_stride')),
    test=dict(
        ann_file=data_root + f'dataset/cross_valid/{str(fold)}/test.json',
        img_prefix=data_root + 'data',
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=20,
            frame_range=[-10,0],
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
total_epochs = 20
evaluation = dict(metric=['bbox'], interval=1)
# load_from = r'/srv/fenster/people/Ningtao/Project/ThyroidNodule/solo-learn/checkpoint/drchong_c04/simclr/offline-2oj5u4n5/simclr-800ep-drchong_modified-offline-2oj5u4n5-ep=983.pth'
# load_from = '/srv/fenster/people/Ningtao/Project/ThyroidNodule/solo-learn/checkpoint/drchong_modified_expand_0606/simclr/offline-0ft5a6c7/simclr-800ep-drchong_modified-offline-0ft5a6c7-ep=999.pth'
load_from = '/srv/fenster/people/Ningtao/Project/ThyroidNodule/solo-learn/checkpoint/drchong_c05_mm/simclr/offline-u7v250ix/mm_ep=469.pth'
