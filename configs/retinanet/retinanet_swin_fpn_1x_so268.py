_base_ = 'retinanet_r50_fpn_1x_coco.py'

# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
pretrained = 'checkpoints/swin_tiny_patch4_window7_224.pth'

model = {
   'backbone': {
      '_delete_': True,
      'type': 'SwinTransformer',
      'embed_dims': 96,
      'depths': [2, 2, 6, 2],
      'num_heads': [3, 6, 12, 24],
      'window_size': 7,
      'mlp_ratio': 4,
      'qkv_bias': True,
      'qk_scale': None,
      'drop_rate': 0.,
      'attn_drop_rate': 0.,
      'drop_path_rate': 0.2,
      'patch_norm': True,
      'out_indices': (0, 1, 2, 3),
      'with_cp': False,
      'convert_weights': True,
      'init_cfg': {
         'type': 'Pretrained',
         'checkpoint': pretrained,
      },
   },
   'neck': {'in_channels': [96, 192, 384, 768]},
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
   'samples_per_gpu': 8,
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
