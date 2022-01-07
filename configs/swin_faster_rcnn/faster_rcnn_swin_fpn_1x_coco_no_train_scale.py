_base_ = 'faster_rcnn_swin_fpn_1x_coco.py'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = {
   'workers_per_gpu': 4,
   'samples_per_gpu': 12,
   'train': {
      'type': 'CustomDataset',
      'img_prefix': 'data/160_164_patches/annotation_patches/images',
      'ann_file': 'data/160_164_patches/dataset.pkl',
      'pipeline': train_pipeline,
   },
}

# TODO: [x] implement training without scaling
# TODO: [o] create 512x512 UnKnoT training dataset
#  -> training patches but transformed annotations are still missing
#  -> script to transform masks to bounding boxes?
# TODO: [ ] implement custom inference script without test augmentation
#  -> https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html
