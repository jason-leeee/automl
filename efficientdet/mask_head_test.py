"""Tests for mask head."""
import os
import tempfile
from absl import logging
import tensorflow.compat.v1 as tf
import efficientdet_arch as legacy_arch
import hparams_config
from keras import efficientdet_keras
from keras import mask_feat_head

SEED = 11111
def test_mask_shape():
    """A function for testing the forward pass of mask head"""

    config = hparams_config.get_efficientdet_config("efficientdet-d0")
    config.max_level = 5
    input = tf.ones([1, 512, 512, 3])
    tf.random.set_random_seed(SEED)
    model = efficientdet_keras.EfficientDetNet(config=config)
    all_feats = model.backbone(input, True)
    print(len(all_feats))
    feats = all_feats[config.min_level:config.max_level + 1]

    for resample_layer in model.resample_layers:
        feats.append(resample_layer(feats[-1], True, None))

    fpn_feats = model.fpn_cells(feats, True)
    print("number of fpn features:", len(fpn_feats))

    for feat_map in fpn_feats:
        print(feat_map.shape)
    mask_head = mask_feat_head.MaskFeatHead(num_classes=config.num_classes,
                                            out_channels=64,
                                            in_channels=16,
                                            start_level=config.min_level,
                                            end_level = config.max_level,
                                            strategy=config.strategy,
                                            is_training_bn=True,
                                            batch_size=1)

    ins_pred = mask_head(fpn_feats)
    print("mask head forward pass!")


if __name__ == "__main__":
    test_mask_shape()
