_base_ = 'retinanet_r50_fpn_1x_coco.py'

# pretrained = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
pretrained = 'checkpoints/resnet50-0676ba61.pth'

model = {
   'backbone': {
      'init_cfg': {
         'type': 'Pretrained',
         'checkpoint': pretrained,
      },
   },
   'bbox_head': {'num_classes': 1},
}

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

dataset_type = 'CustomDataset'
data_root = 'data/'
classes = ('interesting',)
data = {
   'workers_per_gpu': 4,
   'samples_per_gpu': 16,
   'train': {
      'type': 'CustomDataset',
      'img_prefix': 'data/160_164_patches/annotation_patches/images',
      'ann_file': 'data/160_164_patches/dataset_nonzero.pkl',
      'pipeline': train_pipeline,
   },
   'val': {
      'type': 'CustomDataset',
      'img_prefix': 'data/SO268-2/100-1_OFOS-05',
      'classes': classes,
      'ann_file': 'data/SO268-2/target_dataset.pickle',
   },
   'test': {
      'type': 'CustomDataset',
      'img_prefix': 'data/SO268-2/100-1_OFOS-05',
      'classes': classes,
      'ann_file': 'data/SO268-2/target_dataset.pickle',
   },
}

evaluation = {'metric': ['mAP']}
log_config = {'interval': 1}
