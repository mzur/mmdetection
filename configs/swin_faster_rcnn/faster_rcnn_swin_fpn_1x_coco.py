# _base_ = '../swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

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
   'roi_head': {
      'bbox_head': {'num_classes': 1},
   },
}

dataset_type = 'CustomDataset'
data_root = 'data/'
classes = ('interesting',)
data = {
   'workers_per_gpu': 5,
   'samples_per_gpu': 3,
   'train': {
      'type': 'CustomDataset',
      'img_prefix': 'data/SO268-2/160-1_OFOS-11_164-1_OFOS-12',
      'classes': classes,
      'ann_file': 'data/SO268-2/source_dataset.pickle',
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

runner = {'max_epochs': 12}
log_config = {'interval': 1}

evaluation = {'metric': ['mAP']}
# resume_from = 'work_dirs/faster_rcnn_swin_fpn_1x_coco/epoch_1.pth'

# load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
