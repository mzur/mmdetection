_base_ = 'faster_rcnn_swin_fpn_1x_coco.py'

model = {
   'backbone': {
      'frozen_stages': 4,
   },
}

runner = {'max_epochs': 12}
