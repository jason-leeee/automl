# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Training related libraries."""
from concurrent import futures
import math
import os
import re
from absl import logging
import numpy as np
import tensorflow as tf
from scipy import ndimage
import inference
import iou_utils
import utils
from keras import anchors
from keras import efficientdet_keras
import neural_structured_learning as nsl
from keras import util_keras
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper


def _collect_prunable_layers(model):
  """Recursively collect the prunable layers in the model."""
  prunable_layers = []
  for layer in model._flatten_layers(recursive=False, include_self=False):
    # A keras model may have other models as layers.
    if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
      prunable_layers.append(layer)
    elif isinstance(layer, (tf.keras.Model, tf.keras.layers.Layer)):
      prunable_layers += _collect_prunable_layers(layer)

  return prunable_layers


class UpdatePruningStep(tf.keras.callbacks.Callback):
  """Keras callback which updates pruning wrappers with the optimizer step.
  This callback must be used when training a model which needs to be pruned. Not
  doing so will throw an error.
  Example:
  ```python
  model.fit(x, y,
      callbacks=[UpdatePruningStep()])
  ```
  """

  def __init__(self):
    super(UpdatePruningStep, self).__init__()
    self.prunable_layers = []

  def on_train_begin(self, logs=None):
    # Collect all the prunable layers in the model.
    self.prunable_layers = _collect_prunable_layers(self.model)
    self.step = tf.keras.backend.get_value(self.model.optimizer.iterations)

  def on_train_batch_begin(self, batch, logs=None):
    tuples = []

    for layer in self.prunable_layers:
      if layer.built:
        tuples.append((layer.pruning_step, self.step))

    tf.keras.backend.batch_set_value(tuples)
    self.step = self.step + 1

  def on_epoch_end(self, batch, logs=None):
    # At the end of every epoch, remask the weights. This ensures that when
    # the model is saved after completion, the weights represent mask*weights.
    weight_mask_ops = []

    for layer in self.prunable_layers:
      if layer.built and isinstance(layer, pruning_wrapper.PruneLowMagnitude):
        if tf.executing_eagerly():
          layer.pruning_obj.weight_mask_op()
        else:
          weight_mask_ops.append(layer.pruning_obj.weight_mask_op())

    tf.keras.backend.batch_get_value(weight_mask_ops)


class PruningSummaries(tf.keras.callbacks.TensorBoard):
  """A Keras callback for adding pruning summaries to tensorboard.

  Logs the sparsity(%) and threshold at a given iteration step.
  """

  def __init__(self, log_dir, update_freq='epoch', **kwargs):
    if not isinstance(log_dir, str) or not log_dir:
      raise ValueError(
          '`log_dir` must be a non-empty string. You passed `log_dir`='
          '{input}.'.format(input=log_dir))

    super().__init__(
        log_dir=log_dir, update_freq=update_freq, **kwargs)

    log_dir = self.log_dir + '/metrics'
    self._file_writer = tf.summary.create_file_writer(log_dir)

  def _log_pruning_metrics(self, logs, step):
    with self._file_writer.as_default():
      for name, value in logs.items():
        tf.summary.scalar(name, value, step=step)

      self._file_writer.flush()

  def on_epoch_begin(self, epoch, logs=None):
    if logs is not None:
      super().on_epoch_begin(epoch, logs)

    pruning_logs = {}
    params = []
    prunable_layers = _collect_prunable_layers(self.model)
    for layer in prunable_layers:
      for _, mask, threshold in layer.pruning_vars:
        params.append(mask)
        params.append(threshold)

    params.append(self.model.optimizer.iterations)

    values = tf.keras.backend.batch_get_value(params)
    iteration = values[-1]
    del values[-1]
    del params[-1]

    param_value_pairs = list(zip(params, values))

    for mask, mask_value in param_value_pairs[::2]:
      pruning_logs.update({
          mask.name + '/sparsity': 1 - np.mean(mask_value)
      })

    for threshold, threshold_value in param_value_pairs[1::2]:
      pruning_logs.update({threshold.name + '/threshold': threshold_value})

    self._log_pruning_metrics(pruning_logs, iteration)

def update_learning_rate_schedule_parameters(params):
  """Updates params that are related to the learning rate schedule."""
  batch_size = params['batch_size'] * params['num_shards']
  # Learning rate is proportional to the batch size
  params['adjusted_learning_rate'] = (params['learning_rate'] * batch_size / 64)
  steps_per_epoch = params['steps_per_epoch']
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
  params['first_lr_drop_step'] = int(params['first_lr_drop_epoch'] *
                                     steps_per_epoch)
  params['second_lr_drop_step'] = int(params['second_lr_drop_epoch'] *
                                      steps_per_epoch)
  params['total_steps'] = int(params['num_epochs'] * steps_per_epoch)


class StepwiseLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
  """Stepwise learning rate schedule."""

  def __init__(self, adjusted_lr: float, lr_warmup_init: float,
               lr_warmup_step: int, first_lr_drop_step: int,
               second_lr_drop_step: int):
    """Build a StepwiseLrSchedule.

    Args:
      adjusted_lr: `float`, The initial learning rate.
      lr_warmup_init: `float`, The warm up learning rate.
      lr_warmup_step: `int`, The warm up step.
      first_lr_drop_step: `int`, First lr decay step.
      second_lr_drop_step: `int`, Second lr decay step.
    """
    super().__init__()
    logging.info('LR schedule method: stepwise')
    self.adjusted_lr = adjusted_lr
    self.lr_warmup_init = lr_warmup_init
    self.lr_warmup_step = lr_warmup_step
    self.first_lr_drop_step = first_lr_drop_step
    self.second_lr_drop_step = second_lr_drop_step

  def __call__(self, step):
    linear_warmup = (
        self.lr_warmup_init +
        (tf.cast(step, dtype=tf.float32) / self.lr_warmup_step *
         (self.adjusted_lr - self.lr_warmup_init)))
    learning_rate = tf.where(step < self.lr_warmup_step, linear_warmup,
                             self.adjusted_lr)
    lr_schedule = [[1.0, self.lr_warmup_step], [0.1, self.first_lr_drop_step],
                   [0.01, self.second_lr_drop_step]]
    for mult, start_global_step in lr_schedule:
      learning_rate = tf.where(step < start_global_step, learning_rate,
                               self.adjusted_lr * mult)
    return learning_rate


class CosineLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
  """Cosine learning rate schedule."""

  def __init__(self, adjusted_lr: float, lr_warmup_init: float,
               lr_warmup_step: int, total_steps: int):
    """Build a CosineLrSchedule.

    Args:
      adjusted_lr: `float`, The initial learning rate.
      lr_warmup_init: `float`, The warm up learning rate.
      lr_warmup_step: `int`, The warm up step.
      total_steps: `int`, Total train steps.
    """
    super().__init__()
    logging.info('LR schedule method: cosine')
    self.adjusted_lr = adjusted_lr
    self.lr_warmup_init = lr_warmup_init
    self.lr_warmup_step = lr_warmup_step
    self.decay_steps = tf.cast(total_steps - lr_warmup_step, tf.float32)

  def __call__(self, step):
    linear_warmup = (
        self.lr_warmup_init +
        (tf.cast(step, dtype=tf.float32) / self.lr_warmup_step *
         (self.adjusted_lr - self.lr_warmup_init)))
    cosine_lr = 0.5 * self.adjusted_lr * (
        1 + tf.cos(math.pi * tf.cast(step, tf.float32) / self.decay_steps))
    return tf.where(step < self.lr_warmup_step, linear_warmup, cosine_lr)


class PolynomialLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
  """Polynomial learning rate schedule."""

  def __init__(self, adjusted_lr: float, lr_warmup_init: float,
               lr_warmup_step: int, power: float, total_steps: int):
    """Build a PolynomialLrSchedule.

    Args:
      adjusted_lr: `float`, The initial learning rate.
      lr_warmup_init: `float`, The warm up learning rate.
      lr_warmup_step: `int`, The warm up step.
      power: `float`, power.
      total_steps: `int`, Total train steps.
    """
    super().__init__()
    logging.info('LR schedule method: polynomial')
    self.adjusted_lr = adjusted_lr
    self.lr_warmup_init = lr_warmup_init
    self.lr_warmup_step = lr_warmup_step
    self.power = power
    self.total_steps = total_steps

  def __call__(self, step):
    linear_warmup = (
        self.lr_warmup_init +
        (tf.cast(step, dtype=tf.float32) / self.lr_warmup_step *
         (self.adjusted_lr - self.lr_warmup_init)))
    polynomial_lr = self.adjusted_lr * tf.pow(
        1 - (tf.cast(step, dtype=tf.float32) / self.total_steps), self.power)
    return tf.where(step < self.lr_warmup_step, linear_warmup, polynomial_lr)


def learning_rate_schedule(params):
  """Learning rate schedule based on global step."""
  update_learning_rate_schedule_parameters(params)
  lr_decay_method = params['lr_decay_method']
  if lr_decay_method == 'stepwise':
    return StepwiseLrSchedule(params['adjusted_learning_rate'],
                              params['lr_warmup_init'],
                              params['lr_warmup_step'],
                              params['first_lr_drop_step'],
                              params['second_lr_drop_step'])

  if lr_decay_method == 'cosine':
    return CosineLrSchedule(params['adjusted_learning_rate'],
                            params['lr_warmup_init'], params['lr_warmup_step'],
                            params['total_steps'])

  if lr_decay_method == 'polynomial':
    return PolynomialLrSchedule(params['adjusted_learning_rate'],
                                params['lr_warmup_init'],
                                params['lr_warmup_step'],
                                params['poly_lr_power'], params['total_steps'])

  raise ValueError('unknown lr_decay_method: {}'.format(lr_decay_method))


def get_optimizer(params):
  """Get optimizer."""
  learning_rate = learning_rate_schedule(params)
  momentum = params['momentum']
  if params['optimizer'].lower() == 'sgd':
    logging.info('Use SGD optimizer')
    optimizer = tf.keras.optimizers.SGD(
        learning_rate, momentum=momentum)
  elif params['optimizer'].lower() == 'adam':
    logging.info('Use Adam optimizer')
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=momentum)
  else:
    raise ValueError('optimizers should be adam or sgd')

  moving_average_decay = params['moving_average_decay']
  if moving_average_decay:
    # TODO(tanmingxing): potentially add dynamic_decay for new tfa release.
    from tensorflow_addons import optimizers as tfa_optimizers  # pylint: disable=g-import-not-at-top
    optimizer = tfa_optimizers.MovingAverage(
        optimizer, average_decay=moving_average_decay, dynamic_decay=True)
  if params['mixed_precision']:
    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
        optimizer, loss_scale=tf.mixed_precision.experimental.DynamicLossScale(params['loss_scale']))
  return optimizer


class DisplayCallback(tf.keras.callbacks.Callback):
  """Display inference result callback."""

  def __init__(self, sample_image, output_dir, update_freq=None):
    super().__init__()
    image_file = tf.io.read_file(sample_image)
    self.sample_image = tf.expand_dims(
        tf.image.decode_jpeg(image_file, channels=3), axis=0)
    self.executor = futures.ThreadPoolExecutor(max_workers=1)
    self.update_freq = update_freq
    self.output_dir = output_dir

  def set_model(self, model: tf.keras.Model):
    self.train_model = model
    with tf.device('/cpu:0'):
      self.model = efficientdet_keras.EfficientDetModel(config=model.config)
    height, width = utils.parse_image_size(model.config.image_size)
    self.model.build((1, height, width, 3))
    log_dir = os.path.join(self.output_dir, 'test_images')
    self.file_writer = tf.summary.create_file_writer(log_dir)
    self.min_score_thresh = self.model.config.nms_configs['score_thresh'] or 0.4
    self.max_boxes_to_draw = (
        self.model.config.nms_configs['max_output_size'] or 100)

  def on_train_batch_end(self, batch, logs=None):
    if self.update_freq and batch % self.update_freq == 0:
      self.executor.submit(self.draw_inference, batch)

  @tf.function
  def inference(self):
    return self.model(self.sample_image, training=False)

  def draw_inference(self, step):
    self.model.set_weights(self.train_model.get_weights())
    boxes, scores, classes, valid_len = self.inference()
    length = valid_len[0]
    image = inference.visualize_image(
        self.sample_image[0],
        boxes[0].numpy()[:length],
        classes[0].numpy().astype(np.int)[:length],
        scores[0].numpy()[:length],
        label_map=self.model.config.label_map,
        min_score_thresh=self.min_score_thresh,
        max_boxes_to_draw=self.max_boxes_to_draw)

    with self.file_writer.as_default():
      tf.summary.image('Test image', tf.expand_dims(image, axis=0), step=step)


def get_callbacks(params):
  """Get callbacks for given params."""
  ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
      os.path.join(params['model_dir'], 'ckpt'),
      verbose=1,
      save_weights_only=True)
  callbacks = [ckpt_callback]
  if params['model_optimizations'] and 'prune' in params['model_optimizations']:
    prune_callback = UpdatePruningStep()
    prune_summaries = PruningSummaries(
        log_dir=params['model_dir'],
        update_freq=params['iterations_per_loop'],
        profile_batch=2 if params['profile'] else 0)
    callbacks += [prune_callback, prune_summaries]
  else:
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=params['model_dir'], update_freq=params['iterations_per_loop'],
        profile_batch=2 if params['profile'] else 0)
    callbacks.append(tb_callback)
  if params.get('sample_image', None):
    display_callback = DisplayCallback(
        params.get('sample_image', None),
        params['model_dir'],
        params['img_summary_steps'])
    callbacks.append(display_callback)
  return callbacks


class AdversarialLoss(tf.keras.losses.Loss):
  """Adversarial keras loss wrapper"""
  #TODO(fsx950223): WIP
  def __init__(self, adv_config, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.adv_config = adv_config
    self.model = None
    self.loss_fn = None
    self.tape = None
    self.built = False

  def build(self, model, loss_fn, tape):
    self.model = model
    self.loss_fn = loss_fn
    self.tape = tape
    self.built = True

  def call(self, features, y, y_pred, labeled_loss):
    return self.adv_config.multiplier * nsl.keras.adversarial_loss(
        features,
        y,
        self.model,
        self.loss_fn,
        predictions=y_pred,
        labeled_loss=self.labeled_loss,
        gradient_tape=self.tape)


class FocalLoss(tf.keras.losses.Loss):
  """Compute the focal loss between `logits` and the golden `target` values.

  Focal loss = -(1-pt)^gamma * log(pt)
  where pt is the probability of being classified to the true class.
  """

  def __init__(self, alpha, gamma, label_smoothing=0.0, **kwargs):
    """Initialize focal loss.

    Args:
      alpha: A float32 scalar multiplying alpha to the loss from positive
        examples and (1-alpha) to the loss from negative examples.
      gamma: A float32 scalar modulating loss from hard and easy examples.
      label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
      **kwargs: other params.
    """
    super().__init__(**kwargs)
    self.alpha = alpha
    self.gamma = gamma
    self.label_smoothing = label_smoothing

  @tf.autograph.experimental.do_not_convert
  def call(self, y, y_pred):
    """Compute focal loss for y and y_pred.

    Args:
      y: A tuple of (normalizer, y_true), where y_true is the target class.
      y_pred: A float32 tensor [batch, height_in, width_in, num_predictions].

    Returns:
      the focal loss.
    """
    normalizer, y_true = y
    alpha = tf.convert_to_tensor(self.alpha, dtype=y_pred.dtype)
    gamma = tf.convert_to_tensor(self.gamma, dtype=y_pred.dtype)

    # compute focal loss multipliers before label smoothing, such that it will
    # not blow up the loss.
    pred_prob = tf.sigmoid(y_pred)
    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t)**gamma

    # apply label smoothing for cross_entropy for each entry.
    y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # compute the final loss and return
    return alpha_factor * modulating_factor * ce / normalizer


class BoxLoss(tf.keras.losses.Loss):
  """L2 box regression loss."""

  def __init__(self, delta=0.1, **kwargs):
    """Initialize box loss.

    Args:
      delta: `float`, the point where the huber loss function changes from a
        quadratic to linear. It is typically around the mean value of regression
        target. For instances, the regression targets of 512x512 input with 6
        anchors on P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
      **kwargs: other params.
    """
    super().__init__(**kwargs)
    self.huber = tf.keras.losses.Huber(
        delta, reduction=tf.keras.losses.Reduction.NONE)

  @tf.autograph.experimental.do_not_convert
  def call(self, y_true, box_outputs):
    num_positives, box_targets = y_true
    normalizer = num_positives * 4.0
    mask = tf.cast(box_targets != 0.0, num_positives.dtype)
    box_targets = tf.expand_dims(box_targets, axis=-1)
    box_outputs = tf.expand_dims(box_outputs, axis=-1)
    # TODO(fsx950223): remove cast when huber loss dtype is fixed.
    box_loss = tf.cast(self.huber(box_targets, box_outputs),
                       num_positives.dtype) * mask
    box_loss = tf.reduce_sum(box_loss) / normalizer
    return box_loss


class BoxIouLoss(tf.keras.losses.Loss):
  """Box iou loss."""

  def __init__(self, iou_loss_type, min_level, max_level, num_scales,
               aspect_ratios, anchor_scale, image_size, **kwargs):
    super().__init__(**kwargs)
    self.iou_loss_type = iou_loss_type
    self.input_anchors = anchors.Anchors(min_level, max_level, num_scales,
                                         aspect_ratios, anchor_scale,
                                         image_size)

  @tf.autograph.experimental.do_not_convert
  def call(self, y_true, box_outputs):
    anchor_boxes = tf.tile(
        self.input_anchors.boxes,
        [box_outputs.shape[0] // self.input_anchors.boxes.shape[0], 1])
    num_positives, box_targets = y_true
    normalizer = num_positives * 4.0
    mask = tf.cast(box_targets != 0.0, num_positives.dtype)
    box_outputs = anchors.decode_box_outputs(box_outputs, anchor_boxes) * mask
    box_targets = anchors.decode_box_outputs(box_targets, anchor_boxes) * mask
    box_iou_loss = iou_utils.iou_loss(box_outputs, box_targets,
                                      self.iou_loss_type)
    box_iou_loss = tf.reduce_sum(box_iou_loss) / normalizer
    return box_iou_loss


class SOLOLoss(tf.keras.losses.Loss):
  def __init__(self, ins_loss_weight, 
                cate_out_channels, cfg, 
                strides=(4, 8, 16, 32, 64),
                scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                sigma = 0.2,
                num_grids=None):
    """
    Args:
      ins_loss_weight: the weight between focal loss and dice loss
    """
    super().__init__()
    self.scale_ranges = scale_ranges
    self.strides = strides
    self.seg_num_grids = num_grids
    self.cfg = cfg
    self.ins_loss_weight = ins_loss_weight
    self.cate_out_channels = cate_out_channels - 1
    self.sigma = sigma
  
  def dice_loss(self, input, target):
    input = tf.reshape(input, [input.shape[0], -1])
    target = tf.cast(tf.reshape(target, [target.shape[0], -1]), tf.float32)

    a = tf.keras.backend.sum(input * target, 1)
    b = tf.keras.backend.sum(input * input, 1) + 0.001
    c = tf.keras.backend.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1-d

  def single_target(self, gt_bboxes_raw,
                          gt_labels_raw,
                          gt_masks_raw,
                          mask_feat_size):
    """
    Generate training targets for a single image.
    Args:
      gt_bboxes_raw: shape of [num_objects, 4]
      gt_labels_raw: shape of [num_objects, 1]
      gt_masks_raw: shape of [num_object, 1, H, W]
    """                      
    gt_areas = tf.math.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

    ins_label_list = []
    cate_label_list = []
    ins_ind_label_list = []
    grid_order_list = []

    for (lower_bound, upper_bound), stride, num_grid \
          in zip(self.scale_ranges, self.strides, self.seg_num_grids):
      hit_indices = tf.keras.backend.flatten(tf.where(((gt_areas >= lower_bound) and (gt_areas <= upper_bound))))
      num_ins = len(hit_indices)
      ins_label = []
      grid_order = []
      cate_label = tf.zeros([num_grid, num_grid], dtype=tf.int64).numpy()
      ins_ind_label = tf.zeros([num_grid ** 2], dtype=tf.bool).numpy()
      if num_ins == 0:
        ins_label = tf.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=tf.uint8)
        ins_label_list.append(ins_label)
        cate_label_list.append(cate_label)
        ins_ind_label_list.append(ins_ind_label)
        grid_order_list.append([])
        continue
      gt_bboxes = tf.gather(gt_bboxes_raw, hit_indices)
      gt_labels = tf.gather(gt_labels_raw, hit_indices)
      gt_masks = tf.gather_nd(gt_masks_raw, hit_indices)

      half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
      half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

      output_stride = 4
      img_scale = 1. / output_stride
      for seg_mask, gt_label, half_h, half_w in zip(gt_masks, gt_labels, half_hs, half_ws):
        if tf.keras.backend.sum(seg_mask) == 0:
            continue
        # mass center
        upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
        center_h, center_w = ndimage.measurements.center_of_mass(seg_mask.numpy())
        coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
        coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

        # left, top, right, down
        top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
        down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
        left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
        right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

        top = max(top_box, coord_h-1)
        down = min(down_box, coord_h+1)
        left = max(coord_w-1, left_box)
        right = min(right_box, coord_w+1)

        cate_label[top:(down+1), left:(right+1)] = gt_label
        new_w, new_h = int(seg_mask.shape[-2] * float(img_scale) + 0.5), int(seg_mask.shape[-1] * float(img_scale) + 0.5)
        # resize segmask to [batch, h, w, channel]
        seg_mask = tf.image.resize(seg_mask[tf.newaxis, ..., tf.newaxis], [new_w, new_h])
        for i in range(top, down+1):
          for j in range(left, right+1):
            label = int(i * num_grid + j)

            cur_ins_label = tf.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=tf.uint8).numpy()
            cur_ins_label[:seg_mask.shape[1], :seg_mask.shape[2]] = tf.squeeze(seg_mask)
            ins_label.append(cur_ins_label)
            ins_ind_label[label] = True
            grid_order.append(label)
      
      ins_label = tf.stack(ins_label, 0)

      ins_label_list.append(ins_label)
      cate_label_list.append(cate_label)
      ins_ind_label_list.append(ins_ind_label)
      grid_order_list.append(grid_order)
    return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

  @tf.autograph.experimental.do_not_convert  
  def call(self, y_true, y_preds):
    cate_preds, kernel_preds, ins_pred = y_preds
    gt_bbox_list, gt_label_list, gt_mask_list = y_true
    mask_feat_size = ins_pred.shape[:-1]
    print("mask_feat_size:", mask_feat_size)
    ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = util_keras.multi_apply(
                                                                            self.single_target,
                                                                            gt_bbox_list,
                                                                            gt_label_list,
                                                                            gt_mask_list, 
                                                                            mask_feat_size=mask_feat_size)
    # ins
    ins_labels = [tf.concat([ins_labels_level_img
                            for ins_labels_level_img in ins_labels_level], 0)
                  for ins_labels_level in zip(*ins_label_list)]

    print("oders", grid_order_list)

    kernel_preds_list = []
    for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list)):
      kernel_preds_level_list = []
      for kernel_preds_level_img, grid_orders_level_img in zip(kernel_preds_level, grid_orders_level):
        print(kernel_preds_level_img.shape) 
        #print(grid_orders_level_img)
        kernel_preds_level_img = tf.reshape(kernel_preds_level_img, [-1, kernel_preds_level_img.shape[-1]])
        print("kernel_preds_level_img shape:", kernel_preds_level_img.shape)
        grid_orders_level_img = tf.convert_to_tensor(grid_orders_level_img, dtype=tf.int32)
        #print(grid_orders_level_img)
        kernel_preds_level_list.append(tf.gather(kernel_preds_level_img, grid_orders_level_img))
        #print("hello there:", kernel_preds_level)

      kernel_preds_list.append(kernel_preds_level_list)
                 
    # generate masks
    ins_pred = ins_pred
    ins_pred_list = []
    for b_kernel_pred in kernel_preds_list:
      b_mask_pred = []
      for idx, kernel_pred in enumerate(b_kernel_pred):
        print("kernel_pred shape:", kernel_pred.shape)
        if kernel_pred.shape[0] == 0:
          continue
        cur_ins_pred = ins_pred[..., idx]
        print("cur_ins_pred:", idx, cur_ins_pred.shape)
        H, W = cur_ins_pred.shape[-2:]
        N, I = kernel_pred.shape
        print(H, W, N, I)
        cur_ins_pred = cur_ins_pred[tf.newaxis, ..., tf.newaxis] #shape [B, H, W, C]
        kernel_pred = tf.reshape(kernel_pred, [1, 1, -1, I]) #shape [H, W, in, out]
        cur_ins_pred = tf.reshape(tf.nn.conv2d(cur_ins_pred, kernel_pred, strides=1, padding='SAME'), [-1, H, W])
        b_mask_pred.append(cur_ins_pred)
        print("transposed kernel_pred shape:", kernel_pred.shape)
        print("convolved cur_ins_pred shape:", cur_ins_pred.shape)
      if len(b_mask_pred) == 0:
          b_mask_pred = None
      else:
          b_mask_pred = tf.concat(b_mask_pred, 0)
      ins_pred_list.append(b_mask_pred)

    print("check ins_ind labels shape", ins_ind_label_list[0][0].shape)  
 
    ins_ind_labels = [
            tf.concat([tf.keras.backend.flatten(ins_ind_labels_level_img)
                       for ins_ind_labels_level_img in ins_ind_labels_level], 0)
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
    #print(ins_ind_labels)
    flatten_ins_ind_labels = tf.concat(ins_ind_labels, 0)
 
    num_ins = tf.math.reduce_sum(tf.cast(flatten_ins_ind_labels, dtype=tf.uint8))
    
    # dice loss
    loss_ins = []
    for input, target in zip(ins_pred_list, ins_labels):
      if input is None:
        continue
      input = tf.keras.activations.sigmoid(input)
      loss_ins.append(self.dice_loss(input, target))
    loss_ins = tf.math.reduce_mean(tf.stack(loss_ins))
    loss_ins = loss_ins * self.ins_loss_weight
    print("loss_ins:", loss_ins)

    # cate
    cate_labels = [
        tf.concat([tf.keras.backend.flatten(cate_labels_level_img)
                    for cate_labels_level_img in cate_labels_level], axis=0)
        for cate_labels_level in zip(*cate_label_list)
    ]
    print("cate_label shape", cate_labels[0].shape)
    flatten_cate_labels = tf.concat(cate_labels, axis=0)
    print("flattened cate_label shape", cate_labels[0].shape)

    for cate_pred in cate_preds:
      print("cate_pred shape", cate_pred.shape)
    cate_preds = [tf.reshape(cate_pred, [-1, self.cate_out_channels])
        for cate_pred in cate_preds
    ]
    flatten_cate_preds = tf.concat(cate_preds, axis=0)

    return loss_ins
    #loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)
    #return dict(
    #    loss_ins=loss_ins,
    #    loss_cate=loss_cate)


class EfficientDetNetTrain(efficientdet_keras.EfficientDetNet):
  """A customized trainer for EfficientDet.

  see https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    log_dir = os.path.join(self.config.model_dir, 'train_images')
    self.summary_writer = tf.summary.create_file_writer(log_dir)

  def _freeze_vars(self):
    if self.config.var_freeze_expr:
      return [
          v for v in self.trainable_variables
          if not re.match(self.config.var_freeze_expr, v.name)
      ]
    return self.trainable_variables

  def _reg_l2_loss(self, weight_decay, regex=r'.*(kernel|weight):0$'):
    """Return regularization l2 loss loss."""
    var_match = re.compile(regex)
    return weight_decay * tf.add_n([
        tf.nn.l2_loss(v)
        for v in self._freeze_vars()
        if var_match.match(v.name)
    ])

  def _detection_loss(self, cls_outputs, box_outputs, labels, loss_vals):
    """Computes total detection loss.

    Computes total detection loss including box and class loss from all levels.
    Args:
      cls_outputs: an OrderDict with keys representing levels and values
        representing logits in [batch_size, height, width, num_anchors].
      box_outputs: an OrderDict with keys representing levels and values
        representing box regression targets in [batch_size, height, width,
        num_anchors * 4].
      labels: the dictionary that returned from dataloader that includes
        groundtruth targets.
      loss_vals: A dict of loss values.

    Returns:
      total_loss: an integer tensor representing total loss reducing from
        class and box losses from all levels.
      cls_loss: an integer tensor representing total class loss.
      box_loss: an integer tensor representing total box regression loss.
      box_iou_loss: an integer tensor representing total box iou loss.
    """
    # Sum all positives in a batch for normalization and avoid zero
    # num_positives_sum, which would lead to inf loss during training
    precision = utils.get_precision(self.config.strategy, self.config.mixed_precision)
    dtype = precision.split('_')[-1]
    num_positives_sum = tf.reduce_sum(labels['mean_num_positives']) + 1.0
    positives_momentum = self.config.positives_momentum or 0
    if positives_momentum > 0:
      # normalize the num_positive_examples for training stability.
      moving_normalizer_var = tf.Variable(
          0.0,
          name='moving_normalizer',
          dtype=dtype,
          synchronization=tf.VariableSynchronization.ON_READ,
          trainable=False,
          aggregation=tf.VariableAggregation.MEAN)
      num_positives_sum = tf.keras.backend.moving_average_update(
          moving_normalizer_var,
          num_positives_sum,
          momentum=self.config.positives_momentum)
    elif positives_momentum < 0:
      num_positives_sum = utils.cross_replica_mean(num_positives_sum)
    num_positives_sum = tf.cast(num_positives_sum, dtype)
    levels = range(len(cls_outputs))
    cls_losses = []
    box_losses = []
    for level in levels:
      # Onehot encoding for classification labels.
      cls_targets_at_level = tf.one_hot(
          labels['cls_targets_%d' % (level + self.config.min_level)],
          self.config.num_classes, dtype=dtype)

      if self.config.data_format == 'channels_first':
        bs, _, width, height, _ = cls_targets_at_level.get_shape().as_list()
        cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                          [bs, -1, width, height])
      else:
        bs, width, height, _, _ = cls_targets_at_level.get_shape().as_list()
        cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                          [bs, width, height, -1])

      class_loss_layer = self.loss.get(FocalLoss.__name__, None)
      if class_loss_layer:
        cls_loss = class_loss_layer([num_positives_sum, cls_targets_at_level],
                                    cls_outputs[level])
        if self.config.data_format == 'channels_first':
          cls_loss = tf.reshape(
              cls_loss, [bs, -1, width, height, self.config.num_classes])
        else:
          cls_loss = tf.reshape(
              cls_loss, [bs, width, height, -1, self.config.num_classes])
        cls_loss *= tf.cast(
            tf.expand_dims(
                tf.not_equal(
                    labels['cls_targets_%d' % (level + self.config.min_level)],
                    -2), -1),
            dtype)
        cls_loss_sum = tf.clip_by_value(tf.reduce_sum(cls_loss), 0.0, 2.0)
        cls_losses.append(tf.cast(cls_loss_sum, dtype))

      if self.config.box_loss_weight and self.loss.get(BoxLoss.__name__, None):
        box_targets_at_level = (
            labels['box_targets_%d' % (level + self.config.min_level)])
        box_loss_layer = self.loss[BoxLoss.__name__]
        box_losses.append(
            box_loss_layer([num_positives_sum, box_targets_at_level],
                           box_outputs[level]))

    if self.config.iou_loss_type:
      box_outputs = tf.concat([tf.reshape(v, [-1, 4]) for v in box_outputs],
                              axis=0)
      box_targets = tf.concat([
          tf.reshape(
              labels['box_targets_%d' % (level + self.config.min_level)],
              [-1, 4])
          for level in levels
      ], axis=0)
      box_iou_loss_layer = self.loss[BoxIouLoss.__name__]
      box_iou_loss = box_iou_loss_layer([num_positives_sum, box_targets],
                                        box_outputs)
      loss_vals['box_iou_loss'] = box_iou_loss
    else:
      box_iou_loss = 0

    cls_loss = tf.add_n(cls_losses) if cls_losses else 0
    box_loss = tf.add_n(box_losses) if box_losses else 0
    total_loss = (
        cls_loss + self.config.box_loss_weight * box_loss +
        self.config.iou_loss_weight * box_iou_loss)
    loss_vals['det_loss'] = total_loss
    loss_vals['cls_loss'] = cls_loss
    loss_vals['box_loss'] = box_loss
    return total_loss

  def train_step(self, data):
    """Train step.

    Args:
      data: Tuple of (images, labels). Image tensor with shape [batch_size,
        height, width, 3]. The height and width are fixed and equal.Input labels
        in a dictionary. The labels include class targets and box targets which
        are dense label maps. The labels are generated from get_input_fn
        function in data/dataloader.py.

    Returns:
      A dict record loss info.
    """
    images, labels = data
    if self.config.img_summary_steps:
      with self.summary_writer.as_default():
        tf.summary.image('input_image', images)
    with tf.GradientTape() as tape:
      if len(self.config.heads) == 2:
        cls_outputs, box_outputs, seg_outputs = self(images, training=True)
      elif 'object_detection' in self.config.heads:
        cls_outputs, box_outputs = self(images, training=True)
      elif 'segmentation' in self.config.heads:
        seg_outputs, = self(images, training=True)
      total_loss = 0
      loss_vals = {}
      if 'object_detection' in self.config.heads:
        det_loss = self._detection_loss(cls_outputs, box_outputs, labels,
                                        loss_vals)
        total_loss += det_loss
      if 'segmentation' in self.config.heads:
        seg_loss_layer = (
            self.loss[tf.keras.losses.SparseCategoricalCrossentropy.__name__])
        seg_loss = seg_loss_layer(labels['image_masks'], seg_outputs)
        total_loss += seg_loss
        loss_vals['seg_loss'] = seg_loss

      reg_l2_loss = self._reg_l2_loss(self.config.weight_decay)
      loss_vals['reg_l2_loss'] = reg_l2_loss
      total_loss += tf.cast(reg_l2_loss, images.dtype)
      if isinstance(self.optimizer,
                    tf.keras.mixed_precision.experimental.LossScaleOptimizer):
        scaled_loss = self.optimizer.get_scaled_loss(total_loss)
      else:
        scaled_loss = total_loss
    loss_vals['loss'] = total_loss
    loss_vals['learning_rate'] = self.optimizer.learning_rate(
        self.optimizer.iterations)
    trainable_vars = self._freeze_vars()
    scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
    if isinstance(self.optimizer,
                  tf.keras.mixed_precision.experimental.LossScaleOptimizer):
      gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
    else:
      gradients = scaled_gradients
    if self.config.clip_gradients_norm > 0:
      clip_norm = abs(self.config.clip_gradients_norm)
      gradients = [
          tf.clip_by_norm(g, clip_norm) if g is not None else None
          for g in gradients
      ]
      gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
      loss_vals['gradient_norm'] = tf.linalg.global_norm(gradients)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    return loss_vals

  def test_step(self, data):
    """Test step.

    Args:
      data: Tuple of (images, labels). Image tensor with shape [batch_size,
        height, width, 3]. The height and width are fixed and equal.Input labels
        in a dictionary. The labels include class targets and box targets which
        are dense label maps. The labels are generated from get_input_fn
        function in data/dataloader.py.

    Returns:
      A dict record loss info.
    """
    images, labels = data
    if len(self.config.heads) == 2:
      cls_outputs, box_outputs, seg_outputs = self(images, training=False)
    elif 'object_detection' in self.config.heads:
      cls_outputs, box_outputs = self(images, training=False)
    elif 'segmentation' in self.config.heads:
      seg_outputs, = self(images, training=False)
    total_loss = 0
    loss_vals = {}
    if 'object_detection' in self.config.heads:
      det_loss = self._detection_loss(cls_outputs, box_outputs, labels,
                                      loss_vals)
      total_loss += det_loss
    if 'segmentation' in self.config.heads:
      seg_loss_layer = (
          self.loss[tf.keras.losses.SparseCategoricalCrossentropy.__name__])
      seg_loss = seg_loss_layer(labels['image_masks'], seg_outputs)
      total_loss += seg_loss
      loss_vals['seg_loss'] = seg_loss
    reg_l2_loss = self._reg_l2_loss(self.config.weight_decay)
    loss_vals['reg_l2_loss'] = reg_l2_loss
    loss_vals['loss'] = total_loss + tf.cast(reg_l2_loss, images.dtype)
    return loss_vals
