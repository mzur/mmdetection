_base_ = 'yolov3_d53_mstrain-416_273e_coco.py'

# pretrained = 'https://download.openmmlab.com/pretrain/third_party/darknet53-a628ea1b.pth'
pretrained = 'checkpoints/darknet53-a628ea1b.pth'

model = {
   'backbone': {
      'init_cfg': {
         'type': 'Pretrained',
         'checkpoint': pretrained,
      },
   },
   'bbox_head': {'num_classes': 1},
}

dataset_type = 'CustomDataset'
data_root = 'data/'
classes = ('interesting',)
data = {
   'workers_per_gpu': 4,
   'samples_per_gpu': 16,
   'train': {
      'type': 'CustomDataset',
      'img_prefix': 'data/160_164_patches/annotation_patches/images',
      'ann_file': 'data/160_164_patches/dataset.pkl',
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
