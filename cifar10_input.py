# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,file:///C:/Users/80443/AppData/Local/Temp/OneNote/16.0/Exported/%7BD10A82E6-A02C-447D-B1CE-E3BF308C36D0%7D/NT/18/cifar10_eval.py
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 15

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1460
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1


def read_cifar10(filename_queue):
  """Reads读取 and parses解析 examples from CIFAR10 data files.
  Recommendation: if you want N-way read parallelism平行, call this function函数
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue数组 of strings with the filenames文件名 to read from.
  Returns:
    An object representing代表 a single example, with the following fields:
      height: number of rows in the result (32)行数 高
      width: number of columns in the result (32)列数 宽
      depth: number of color channels in the result (3)色彩通道 深度
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()
  #定义一个CIFAR10Record的空类

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 15
  result.width = 15
  result.depth = 1
  image_bytes = result.height * result.width * result.depth
  #image_bytes=32*32*3
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes*4 + image_bytes*4

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  
  #读取bin文件
  """
  这里的images的存储字节数是固定，用tf.FixedLengthRecordReader创建reader
  cifar10的数据没有header以及footer，我们以后的数据未知
  """
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  #返回从filename_queue得到的（key,value)，key跟value是string的tensor
  result.key, value = reader.read(filename_queue)
  """
   返回从filename_queue中读取的(key, value)对，key和value都是字符串类型的tensor，并且当队列中的某一个文件读完成时，该文件名会dequeue 
   value应该存储的是图像的信息
  """
  # Convert from a string to a vector of uint8 that is record_bytes long.
  #从string转化为uint8记录record_bytes的long
  record_bytes = tf.decode_raw(value, tf.int32)

  # The first bytes represent the label标签, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
  """
  从读取出来的value中截取出标签的信息存放在result.label
  """
  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  """
  depth_major=[resut.depth,result.height,result.width]
  result.depth为图像的深度内容
  result.height为图像的高的内容
  result.width为图像的宽内容
  """
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])
 
  """
 result.uint8image=[result.height,result.width,result.depth]
  """

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size,n):
  """Construct distorted input for CIFAR training using the Reader ops.
  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  """
  filename_queue=/tmp/cifar10_data/cifar-10-batches-bin-data-batch-1.bin
  /tmp/cifar10_data/cifar-10-batches-bin-data-batch-2.bin
  /tmp/cifar10_data/cifar-10-batches-bin-data-batch-3.bin
  /tmp/cifar10_data/cifar-10-batches-bin-data-batch-4.bin
  /tmp/cifar10_data/cifar-10-batches-bin-data-batch-5.bin
  /tmp/cifar10_data/cifar-10-batches-bin-data-batch-6.bin
  """
  filenames = [os.path.join(data_dir, 'data%d.bin' % i)
               for i in xrange(n,n+1)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  """
  reshaped_image=[image.height,image.width,image.depth] float 32格式
  IMAGE_SIZE=24
  """
  #height = IMAGE_SIZE
 # width = IMAGE_SIZE

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  #distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
  """
  随机剪下[24,24,3]的图像
  """
  # Randomly flip the image horizontally.
  #distorted_image = tf.image.random_flip_left_right(distorted_image)
  """
  图像随机水平翻转
  """

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  # NOTE: since per_image_standardization zeros the mean and makes
  # the stddev unit, this likely has no effect see tensorflow#1458.
  """
  图像的亮度，对比度，随机变化
  """
  #distorted_image = tf.image.random_brightness(distorted_image,
     #                                          max_delta=63)
 # distorted_image = tf.image.random_contrast(distorted_image,
     #                                        lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  """
  标准化后的图片
  """
  float_image = tf.image.per_image_standardization(reshaped_image)
  #float_image=reshaped_image

     
  # Set the shapes of tensors.
  float_image.set_shape([15, 15, 1])
  
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)
 
  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(eval_data, data_dir, batch_size,n):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test%d.bin' %n)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         height, width)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 1])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)