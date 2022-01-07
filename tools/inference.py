import asyncio
import torch
from mmdet.apis import init_detector, async_inference_detector
from mmdet.utils.contextmanagers import concurrent
import sys
import mmcv
import os.path
import csv

# Usage example: find data/SO268-2/100-1_OFOS-05/ | shuf | head -n 10 | xargs python3 tools/inference.py work_dirs/faster_rcnn_swin_fpn_1x_coco/faster_rcnn_swin_fpn_1x_coco.py work_dirs/faster_rcnn_swin_fpn_1x_coco/latest.pth results.csv
#
# sed 's/^/data\/SO268-2\/100-1_OFOS-05\//' data/SO268-2/target_images.csv | xargs python3 tools/inference.py work_dirs/faster_rcnn_swin_fpn_1x_coco/faster_rcnn_swin_fpn_1x_coco.py work_dirs/faster_rcnn_swin_fpn_1x_coco/latest.pth results.csv

config_file = sys.argv[1]
checkpoint_file = sys.argv[2]
results_file = sys.argv[3]
image_paths = sys.argv[4:]

async def main():
    device = 'cuda:0'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    # queue is used for concurrent inference of multiple images
    streamqueue = asyncio.Queue()
    # queue size defines concurrency level
    streamqueue_size = 3

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device=device))

    async def detect(path):
        async with concurrent(streamqueue):
            img = mmcv.imread(path)
            return await async_inference_detector(model, img)

    tasks = [
        asyncio.create_task(detect(path)) for path in image_paths
    ]

    return await asyncio.gather(*tasks)

results = asyncio.run(main())

id_counter = 1

def bbox_to_circle(bbox):
   x1, y1, x2, y2, _ = bbox
   r = round(max(x2 - x1, y2 - y1), 2)
   x = round((x1 + x2) / 2, 2)
   y = round((y1 + y2) / 2, 2)

   return x, y, r

with open(results_file, 'w') as file:
   writer = csv.writer(file)
   writer.writerow(['id', 'points', 'filename', 'label_id'])

   for filename, result in zip(image_paths, results):
      for bbox in result[0]:
         x, y, r = bbox_to_circle(bbox)
         points = '[{:.2f},{:.2f},{:.2f}]'.format(x, y, r)
         writer.writerow([id_counter, points, os.path.basename(filename), 'NULL'])
         id_counter += 1
