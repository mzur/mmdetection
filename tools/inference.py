import asyncio
import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.utils.contextmanagers import concurrent
import sys
import mmcv
import os.path
import csv

# Usage example: find data/SO268-2/100-1_OFOS-05/ | shuf | head -n 10 | xargs python3 tools/inference.py work_dirs/faster_rcnn_swin_fpn_1x_coco/faster_rcnn_swin_fpn_1x_coco.py work_dirs/faster_rcnn_swin_fpn_1x_coco/latest.pth results.csv
#
# sed 's/^/data\/SO268-2\/100-1_OFOS-05\//' data/SO268-2/target_images.csv \
#    | xargs python3 tools/inference.py work_dirs/faster_rcnn_swin_fpn_1x_coco/faster_rcnn_swin_fpn_1x_coco.py \
#       work_dirs/faster_rcnn_swin_fpn_1x_coco/latest.pth \
#       results.csv

config_file = sys.argv[1]
checkpoint_file = sys.argv[2]
results_file = sys.argv[3]
image_paths = sys.argv[4:]

id_counter = 1

def bbox_to_circle(bbox):
   x1, y1, x2, y2, _ = bbox
   r = round(max(x2 - x1, y2 - y1), 2)
   x = round((x1 + x2) / 2, 2)
   y = round((y1 + y2) / 2, 2)

   return x, y, r

device = 'cuda:0'
test_cfg = {
   'rpn': {
      'nms_pre': 1000,
      'max_per_img': 1000,
      'nms': {'type': 'nms', 'iou_threshold': 0.7},
      'min_bbox_size': 0
   },
   'rcnn': {
      'score_thr': 0.05,
      # 'score_thr': 0.70,
      'nms': {'type': 'nms', 'iou_threshold': 0.5},
      'max_per_img': 100
   },
}
# Disable scaling/augmentation during inference.
test_pipeline = [
   {'type': 'LoadImageFromFile'},
   {
      'type': 'MultiScaleFlipAug',
      'scale_factor': 1,
      'flip': False,
      'transforms': [
         {
            'type': 'Normalize',
            'mean': [123.675, 116.28, 103.53],
            'std': [58.395, 57.12, 57.375],
            'to_rgb': True,
         },
         {'type': 'Pad', 'size_divisor': 32},
         {'type': 'ImageToTensor', 'keys': ['img']},
         {'type': 'Collect', 'keys': ['img']},
      ]
   },
]
cfg_options = {
   'model': {
      'test_cfg': test_cfg,
   },
   'data': {
      'test': {'pipeline': test_pipeline},
   },
}
model = init_detector(config_file, checkpoint=checkpoint_file, device=device, cfg_options=cfg_options)

prog_bar = mmcv.ProgressBar(len(image_paths))

with open(results_file, 'w') as file:
   writer = csv.writer(file)
   writer.writerow(['id', 'points', 'filename', 'label_id'])

   for path in image_paths:
      prog_bar.update()
      result = inference_detector(model, path)
      for bbox in result[0]:
         x, y, r = bbox_to_circle(bbox)
         points = '[{:.2f},{:.2f},{:.2f}]'.format(x, y, r)
         writer.writerow([id_counter, points, os.path.basename(path), 'NULL'])
         id_counter += 1
