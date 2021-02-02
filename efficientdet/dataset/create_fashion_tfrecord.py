import collections
import hashlib
import io
import json
import multiprocessing
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import PIL.Image as Image

from pycocotools import mask
import tensorflow as tf
from dataset import label_map_util
from dataset import tfrecord_util

flags.DEFINE_string('image_root', '', 'Directory containing images.')
flags.DEFINE_string('anno_root', '', 'Directory containing annotations.')
flags.DEFINE_string('output_file_prefix', '/tmp/train', 'Path to output file')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')
flags.DEFINE_integer('num_threads', None, 'Number of threads to run.')
FLAGS = flags.FLAGS

#json_name =  'test_fashion_data/anno/' + str(num).zfill(6)+'.json'
#image_name = 'test_fashion_data/image/' + str(num).zfill(6)+'.jpg'

def create_tf_example(image_root, anno_root, image_id):
    image_file_name = str(image_id).zfill(6)+'.jpg'
    image_path = os.path.join(image_root, image_file_name)
    anno_file_name = str(image_id).zfill(6)+'.json'
    anno_path = os.path.join(anno_root, anno_file_name)
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
      encoded_jpg = fid.read()
      #encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(image_path)
    width, height = image.size
    key = hashlib.sha256(encoded_jpg).hexdigest()
    feature_dict = {
        'image/height':
            tfrecord_util.int64_feature(height),
        'image/width':
            tfrecord_util.int64_feature(width),
        'image/filename':
            tfrecord_util.bytes_feature(image_file_name.encode('utf8')),
        'image/source_id':
            tfrecord_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            tfrecord_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            tfrecord_util.bytes_feature(encoded_jpg),
    }
    with open(anno_path, 'r') as f:
        anno = json.loads(f.read())
    # load item annotations
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    cat_ids = []
    style = []
    cate_names = []
    occlusion = []
    zoom_in = []
    scale = []
    viewpoint = []
    encoded_mask_png = []
    object_landmarks = []
    total_num_points = []
    for i in anno:
      if i == 'source' or i=='pair_id':
          continue
      else:
          points = np.zeros(294 * 3)
          box = anno[i]['bounding_box']
          xmin.append(float(box[0]))
          ymin.append(float(box[1]))
          xmax.append(float(box[2]))
          ymax.append(float(box[3]))

          cat_ids.append(anno[i]['category_id'])
          cate_names.append(anno[i]['category_name'].encode('utf-8'))
          style.append(anno[i]['style'])
          occlusion.append(anno[i]['occlusion'])
          zoom_in.append(anno[i]['zoom_in'])
          scale.append(anno[i]['scale'])
          viewpoint.append(anno[i]['viewpoint'])
          # encode to binary mask
          run_len_encoding = mask.frPyObjects(anno[i]['segmentation'], height, width)
          binary_mask = mask.decode(run_len_encoding)
          binary_mask = np.amax(binary_mask, axis=2)
          pil_image = Image.fromarray(binary_mask)
          output_io = io.BytesIO()
          pil_image.save(output_io, format='PNG')
          encoded_mask_png.append(output_io.getvalue())

          # load landmarks
          landmarks = anno[i]['landmarks']

          points_x = landmarks[0::3]
          points_y = landmarks[1::3]
          points_v = landmarks[2::3]
          points_x = np.array(points_x)
          points_y = np.array(points_y)
          points_v = np.array(points_v)

          cat = anno[i]['category_id']
          if cat == 1:
              for n in range(0, 25):
                  points[3 * n] = points_x[n]
                  points[3 * n + 1] = points_y[n]
                  points[3 * n + 2] = points_v[n]
          elif cat ==2:
              for n in range(25, 58):
                  points[3 * n] = points_x[n - 25]
                  points[3 * n + 1] = points_y[n - 25]
                  points[3 * n + 2] = points_v[n - 25]
          elif cat ==3:
              for n in range(58, 89):
                  points[3 * n] = points_x[n - 58]
                  points[3 * n + 1] = points_y[n - 58]
                  points[3 * n + 2] = points_v[n - 58]
          elif cat == 4:
              for n in range(89, 128):
                  points[3 * n] = points_x[n - 89]
                  points[3 * n + 1] = points_y[n - 89]
                  points[3 * n + 2] = points_v[n - 89]
          elif cat == 5:
              for n in range(128, 143):
                  points[3 * n] = points_x[n - 128]
                  points[3 * n + 1] = points_y[n - 128]
                  points[3 * n + 2] = points_v[n - 128]
          elif cat == 6:
              for n in range(143, 158):
                  points[3 * n] = points_x[n - 143]
                  points[3 * n + 1] = points_y[n - 143]
                  points[3 * n + 2] = points_v[n - 143]
          elif cat == 7:
              for n in range(158, 168):
                  points[3 * n] = points_x[n - 158]
                  points[3 * n + 1] = points_y[n - 158]
                  points[3 * n + 2] = points_v[n - 158]
          elif cat == 8:
              for n in range(168, 182):
                  points[3 * n] = points_x[n - 168]
                  points[3 * n + 1] = points_y[n - 168]
                  points[3 * n + 2] = points_v[n - 168]
          elif cat == 9:
              for n in range(182, 190):
                  points[3 * n] = points_x[n - 182]
                  points[3 * n + 1] = points_y[n - 182]
                  points[3 * n + 2] = points_v[n - 182]
          elif cat == 10:
              for n in range(190, 219):
                  points[3 * n] = points_x[n - 190]
                  points[3 * n + 1] = points_y[n - 190]
                  points[3 * n + 2] = points_v[n - 190]
          elif cat == 11:
              for n in range(219, 256):
                  points[3 * n] = points_x[n - 219]
                  points[3 * n + 1] = points_y[n - 219]
                  points[3 * n + 2] = points_v[n - 219]
          elif cat == 12:
              for n in range(256, 275):
                  points[3 * n] = points_x[n - 256]
                  points[3 * n + 1] = points_y[n - 256]
                  points[3 * n + 2] = points_v[n - 256]
          elif cat == 13:
              for n in range(275, 294):
                  points[3 * n] = points_x[n - 275]
                  points[3 * n + 1] = points_y[n - 275]
                  points[3 * n + 2] = points_v[n - 275]
          num_points = len(np.where(points_v > 0)[0])
          total_num_points.append(num_points)
          object_landmarks.append(points.tolist())
    
    feature_dict.update({
        'image/obj_source':
            tfrecord_util.bytes_feature(anno['source'].encode('utf-8')),
        'image/pair_id':
            tfrecord_util.int64_feature(anno['pair_id']),
        'image/object/bbox/xmin':
            tfrecord_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            tfrecord_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            tfrecord_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            tfrecord_util.float_list_feature(ymax),
        'image/object/class/text':
            tfrecord_util.bytes_list_feature(cate_names),
        'image/object/class/label':
            tfrecord_util.int64_list_feature(cat_ids),            
        'image/object/occlusion':
            tfrecord_util.int64_list_feature(occlusion),
        'image/object/style':
            tfrecord_util.int64_list_feature(style),
        'image/object/zoom_in':
            tfrecord_util.int64_list_feature(zoom_in),
        'image/object/scale':
            tfrecord_util.int64_list_feature(scale),
        'image/object/viewpoint':
            tfrecord_util.int64_list_feature(viewpoint),
        'image/pnject/num_keypoints':
            tfrecord_util.int64_list_feature(total_num_points),    
        'image/object/mask':
            tfrecord_util.bytes_list_feature(encoded_mask_png)
    }) # TODO: include viewpoint features
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example

def _pool_create_tf_example(args):
  return create_tf_example(*args)

def _create_tf_record_from_coco_annotations(image_root,
                                            anno_root,
                                            num_images,
                                            output_path,
                                            num_shards):
  """Loads COCO annotation json files and converts to tf.Record format.

  Args:
    image_root: Directory containing the image files.
    anno_root: Directory containing the annotation files.
    num_images: Total number of images in the directory.
    output_path: Path to output tf.Record file.
    num_shards: Number of output files to create.
  """

  logging.info('writing to output path: %s', output_path)
  writers = [
      tf.io.TFRecordWriter(output_path + '-%06d-of-%06d.tfrecord' %
                                  (i, num_shards)) for i in range(num_shards)
  ]
  pool = multiprocessing.Pool(FLAGS.num_threads)

  for idx, (_, tf_example) in enumerate(
      pool.imap(
          _pool_create_tf_example,
          [(image_root, anno_root, img_id) for img_id in range(1, num_images+1)])):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, num_images)

    writers[idx % num_shards].write(tf_example.SerializeToString())

  pool.close()
  pool.join()

  for writer in writers:
    writer.close()

  logging.info('Finished writing')


def main(_):
  assert FLAGS.image_root, '`image_root` missing.'
  assert FLAGS.anno_root, '`anno_root` missing.'

  directory = os.path.dirname(FLAGS.output_file_prefix)
  if not tf.io.gfile.isdir(directory):
    tf.io.gfile.mkdir(directory)

  _create_tf_record_from_coco_annotations(FLAGS.image_root, FLAGS.anno_root,
                                          18, FLAGS.output_file_prefix,
                                          FLAGS.num_shards)


if __name__ == '__main__':
  app.run(main)