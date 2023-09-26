_base_ = [
    '../_base_/models/faster_rcnn_r50_dc5.py',
    '../_base_/datasets/coco_vid_detection_fgfa_drchong_modified.py',
    '../_base_/default_runtime.py'
]
desp='after poly updated the annotation March 18'
model = dict(
    type='SELSA',
    detector=dict(
        roi_head=dict(
            type='AFPRoIHeadBoth',
            aggregator=dict(
                type='SelsaIoUAggregator',
                in_channels=1024,
                num_attention_blocks=16,
                filter_component=[] ,
                aggre_component=['IoU'],
                aggre_factor=[1],
                aux_factor=0.3,
                mode='multiply',
            ),
            bbox_head=dict(
                type='DecupleBBoxHead',
                num_shared_fcs=2,
                num_classes=1,
            ),
                        
        ),
        train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=6000,
            max_per_img=600,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=6000,
                max_per_img=300,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0001,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100,
                mask_thr_binary=0.5)),
        ),    
    )

# dataset settings
fold = 3
data_root = '/path/to/data/root'
data = dict(
    workers_per_gpu=4,
    train=dict(
        ann_file=data_root + f'/path/to/anno/{str(fold)}/train.json',
        img_prefix=data_root + 'data',
        ref_img_sampler=dict(
            num_ref_imgs=8,
            frame_range=21)
    ),
    val=dict(
        ann_file=data_root + f'/path/to/anno/{str(fold)}/val.json',
        img_prefix=data_root + 'data',
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=20,
            frame_range=[-10,0],
            method='test_with_adaptive_stride')),
    test=dict(
        ann_file=data_root + f'/path/to/anno/{str(fold)}/test.json',
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
