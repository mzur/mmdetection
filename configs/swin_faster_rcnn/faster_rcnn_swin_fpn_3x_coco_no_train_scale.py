_base_ = 'faster_rcnn_swin_fpn_1x_coco_no_train_scale.py'

optimizer = {
   '_delete_': True,
   'type': 'AdamW',
   'lr': 0.0001,
   'betas': (0.9, 0.999),
   'weight_decay': 0.05,
   'paramwise_cfg': {
      'custom_keys': {
         'absolute_pos_embed': {'decay_mult': 0.},
         'relative_position_bias_table': {'decay_mult': 0.},
         'norm': {'decay_mult': 0.},
      },
   },
}

lr_config = {'warmup_iters': 1000, 'step': [8, 11]}
runner = {'max_epochs': 36}
