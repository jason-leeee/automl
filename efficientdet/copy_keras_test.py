"""Tests for efficientdet_keras."""
import os
import tempfile
from absl import logging
import tensorflow.compat.v1 as tf
import efficientdet_arch as legacy_arch
import hparams_config
from keras import efficientdet_keras
from keras import train_lib

SEED = 111111



def test_solo_head():
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
    config.max_level = 5
    input = tf.ones([1, 512, 512, 3])
    tf.random.set_random_seed(SEED)
    model = efficientdet_keras.EfficientDetNet(config=config)
    all_feats = model.backbone(input, True)
    feats = all_feats[config.min_level:config.max_level + 1]

    for resample_layer in model.resample_layers:
        feats.append(resample_layer(feats[-1], True, None))

    fpn_feats = model.fpn_cells(feats, True)
    print("number of fpn features:", len(fpn_feats))

    solo_head = efficientdet_keras.SOLOv2Head(
                10,
                config.strategy,
                config.data_format,
                config.is_training_bn,
                num_filters=64,
                stacked_convs=4,
                strides=(4, 8, 16, 32, 64),
                base_edge_list=(16, 32, 64, 128, 256),
                num_grids=[4,6,8,10,12])
    cate_preds, kernel_preds = solo_head(fpn_feats)
    print("pass shape")
    return cate_preds, kernel_preds

def test_solo_single_target():
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
    tf.random.set_random_seed(SEED)
    c = tf.constant([[1.0, 12.0, 24.0, 36.0]])
    d = tf.constant([1.0])
    box_targets = c*d
    cls_targets = tf.ones((1,1))
    image_masks = tf.ones((1, 1, 128, 128))

    solo_loss = train_lib.SOLOLoss(10, 10, config, num_grids=[4,6,8,10,12])

    print("seg map shape", image_masks.shape)
    ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = solo_loss.single_target(
                        box_targets, cls_targets,
                        image_masks, [128,128])
    #print(ins_label_list)
    #print(cate_label_list) 
    #print(ins_ind_label_list) 
    #print(grid_order_list)
    print("target generation forward pass")
    return ins_ind_label_list, cate_label_list, ins_ind_label_list, grid_order_list

def test_solo_loss(cate_preds, kernel_preds):
    ins_preds = tf.ones((128, 128, 64))
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
    tf.random.set_random_seed(SEED)
    c = tf.constant([[1.0, 12.0, 24.0, 36.0]])
    d = tf.constant([1.0])
    box_targets = [c*d]
    cls_targets = [tf.ones((1,1))]
    image_masks = [tf.ones((1, 1, 128, 128))]

    solo_loss = train_lib.SOLOLoss(10, 10, config, num_grids=[4,6,8,10,12])

    y_preds = (cate_preds, kernel_preds, ins_preds)
    y_true = (box_targets, cls_targets, image_masks)
    solo_loss(y_true, y_preds)
    print("pass solo loss")
    
def test_focal_loss(cate_labels_list, cate_preds):
    alpha = 0.25
    gamma = 1.5
    levels = len(cate_preds)
    total_loss = 0
    for level in range(levels):
        print(len(cate_labels_list[level]))
        print(len(cate_preds[level]))
        focal_loss_v2 = train_lib.FocalLoss(
            alpha, gamma)

        cate_pred = tf.concat(cate_preds[level],0)
        labels = tf.stack(cate_labels_list[level],0)
        print(cate_pred.shape, labels.shape)
        labels = tf.one_hot(labels, 9)
        loss = focal_loss_v2([2, labels], cate_pred) 
        total_loss += loss   
    print(total_loss)    
    return total_loss

  

if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  cate_preds, kernel_preds = test_solo_head()
  for kernel_pred in kernel_preds:
      print("kernel shapes:", kernel_pred.shape)
  _, cate_labels_list, _, _ = test_solo_single_target()
  test_solo_loss(cate_preds, kernel_preds)
  test_focal_loss(cate_labels_list, cate_preds)