# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Builds the CIFAR-10 network.
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 100.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.95  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
      x=conv1,第一个特征图
    x: Tensor
  Returns:
    nothing
    创建一个特征图的summaries
    {
    提供特征图的直方图的总结可视化
    提供特征图稀疏测量的总结可视化
    }
    可以通过tensorboard去实现可视化
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  """
  x.op.name='conv1'
  TOWER_NAME='tower'
  经过re.sub()函数后得到tensor_name是字符串'conv1'
  """
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  生成一个有权值衰减的初始化变量
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  权值衰减的值只添加规定的数值
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  """
  dtype =tf.float32
  在cpu上创建变量
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  """
  var 是一个5*5*3*64的四维向量，每一个值是通过标准差为0.05的正态分布取的随机数。dtype=float.32
  wd=0;
  wd!=None;
  进入if判断
  """
  if wd is not None:
      '''
          通过L2范量去求这个四维张量的误差。
          然后以误差，权值衰减值，name作为参数，传入tf.multiply函数中，求得weight_decay
          误差*wd传入weigth_decay中，此处weight_decay=0.0
      '''
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      '''
            以'losses'以及0.0作为参数传入函数add_to_collection中
            把变量weigth_decay这个值传入集合'losses'中，这是一个列表
            <1>现在losses列表中有一个变量 weigth_decay，值为0.0
      '''
      tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs(n):
  """Construct distorted input for CIFAR training using the Reader ops.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size,n=n)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data,n):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size,n=n)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inference(images):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  """
  tf.variable_scope('conv1') as scope：生成一个name为con1的op对象
  给在with里面定义的变量加了一个前缀"conv1/"
  """
  with tf.variable_scope('conv1') as scope:
      #生成一个卷积核
      """
       _variable_with_weight_decay为变量kernel添加权值衰减
        _variable_with_weight_decay变量生成函数，与上面的variable_scope差不多。
        生成的制定标准差的正态分布变量
      """
      kernel = _variable_with_weight_decay('weights',                                       
                                         shape=[5,5,1,64], #shape=[5, 5, 3, 64],#一个5x5的卷积核，64代表输出通道
                                         stddev=5e-2, #标准差 0.05
                                         wd=0.0)
      """
       kernel一个四维向量，里面的值为随机值
       [5,5,3,64]
       images[128,24,24,3]
       按[1,1,1,1]移则生成的特征地图为[128,24,24,64]
       conv保存着128张图片，每张图片有64张24*24的特征图
       conv[128,24,24,64]
       这个特征图是二维的24*24的矩阵
       矩阵中的值就是5*5以24*24中的每一个像素值为中心，然后匹配，无法匹配的为0，
       然后相乘（三个通道都这样），最后把每一个相乘的值相加
       放在24*24的第一个像素值，依次类推，得到了第一张24*24的feature map的所有值
       然后重复操作，换矩阵，得到另外63张 feature map
       最后再换图片，有128张图片 
      """
      conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      """
      传入参数,name='biases'，shape=[64]一维向量，以及一个初始化常数0
      进入函数_variable_on_cpu在cpu上创建变量
      biases,shape=[64],每一个值为0.0
      """
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
      """
      conv中的像素值与biases中的值相加，每次相加64个值，即conv[][][][0-63]与biases[0-63]
      相加，知道conv每一个像素值都加一次0.0.然后生成一个新的四维张量,pre_activation（预激活)
      中的值依然与conv一样。就是一个更新了误差后的特征图四维张量。
      """
      pre_activation = tf.nn.bias_add(conv, biases)
      """
      计算误差之后的pre_activation通过函数tf.nn.relu();
      比较了特征图中的每一个像素值，与0相比，若比0小则用0代替这个值。
      得到一个新的[128][24][24][64]特征图集conv1。，name=conv1.
      """
      conv1 = tf.nn.relu(pre_activation, name=scope.name)
      """
      实现特征图的直方图，稀疏测量的可视化。
      """
      _activation_summary(conv1)

  # pool1
  """
  池化
  tf.nn.max_pool()函数
  参数：特征图：[128][24][24][64]
  池化窗口的大小，[1,3,3,1],每次都只对一张图片一个通道做池化，这两个维度都设为1
  维度上滑动的步长为[1,2,2,1]
  padding有'SAME'和'VALID'类型与卷积相同
  这里的池化操作与卷积操作相同，不过是是把[][1~3][1~3][]每次9个值，取最大的值
  如[][1][1][] [][2][1][]
   [][2][1][] [][2][2][]
   取最大值放在pool1[][1][1][]中，算完后会去算下一个特征图，算完64个特征图后，
   算128图片的64个特征图
   因为滑步步长为[1,2,2,1]
   pool1的shape为[128][12][12][64]
   每一个特征图的尺寸经过池化由24*24变为12*12
  """
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  """
  归一化
  函数tf.nn.lrn()
  通过函数，修改四维向量中的值，大的值变得更大，小的值变得更小。（就是使特征值更加明显）
  但也有人说，这部对算法的提高没什么用，还增加了运算时间。
  """
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    #conv中的值加了0.1
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2,2, 1], padding='SAME', name='pool2')
  
  # local3
  #这一首操作的op对象的name为local3
  """
  pool2存着经过再次卷积，再次池化
  shape=[128,6,6,64]
  Tensor("pool2:0", shape=(128, 6, 6, 64), dtype=float32)
  """
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    """
    把pool2转化矩阵的规模，由四维[128,6,6,64]]转化为二维[128,2304]
    即后面三维[6,6,64]转化为一维[2304];
    即pool2[0][0][0][0]=reshape[0][0]
    pool2[0][0][0][1]=reshape[0][1]
    pool2[0][0][1][0]=reshape[0][64]
    """
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    """
    reshape.get_shape()
    获取到reshape的shape();即【128,2304】
    reshape.get_shape()[1]=2304,
    reshape.get_shape()[0]=128;
    reshape.get_shape()[1].value会得到int 2304
    print(dim)=2304
    """
    dim = reshape.get_shape()[1].value
    """
    stddev=0.04，取卷积核的时候是0.05.然后384？代表的是什么？
    wd=0.04,之前权值衰减值为0.
    0.04怎么确定的?
    <tf.Variable '123:0' shape=(2034, 384) dtype=float32_ref>
    [[-0.05525928  0.02123396  0.00827456 ..., -0.00599061  0.00437567
      -0.05154598]
    [-0.07091329 -0.03093145  0.01149747 ..., -0.07586134 -0.05107301
     0.02299047]
    [-0.04179331  0.00115883 -0.04171174 ...,  0.00422807 -0.04589354
     -0.034286  ]
    ..., 
    [-0.01435822 -0.05352353 -0.0490728  ...,  0.04257048 -0.01741191
     0.0446245 ]
    [-0.03611467 -0.03791223  0.03212654 ..., -0.0402477  -0.03351888
     0.00435678]
    [ 0.06248999  0.07077585  0.00940885 ..., -0.02867841  0.00921156
     0.07232577]]
    weitghts根据随机数生成出来的.以及
    根据生成的随机数，获取一个L2范量，这个值为483左右，每次都跟随机数的取值有关。
    并且把483跟传入的wd0.004相乘，大概是1.93左右传入losses集合中
    weights的为什么是[2034,384]
    wd为什么是0.04
    """
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    """
    biases[0-383]=0.1
    tf.matmul(reshape, weights),a
    一个[128,2034]的矩阵跟一个[2034,384]矩阵相乘。得到一个[128,384]的值
    矩阵相乘的意义：把代表一张图片的所有特征值 2034个值分布与随机生成的2034个随机数相乘后相加得到一个值，
    重复上面的操作384次，得到384个值，取代了图片的特征值，把原先图片的2034个特征值缩小为384个。
    这[128,384]每一个值都+0.1。然后这些值比0小的付0，比0大的维持不变。
    
    """
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
     """
     随机生成一个[384,192]的矩阵，384与local3的二维上限有关
     操作跟local3操作差不多。
     压入losses集合的值根据随机生成的矩阵得到L2范量为45左右，然后乘上wd=0.04；一个大概是0.18的值进入集合。
     """
     weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
     biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
     local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
     _activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    """
    为什么生成随机数的stdeev换成了1/192.0
    经过两层全连接后，不再压权值递减值进入losses集合中，操作一样生成一个[192,10]的矩阵
    10是我们自己定义的，分别对应10个特征图的值。
    矩阵相乘后，这一步没有再用激活函数。
    得到一个[128,10]的神经元，就是每张图片有10个神经元输出，10个值分别对应10个label的值。
    """
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  """
  一堆有128张图片，每张图片都有一个预测交叉熵
  求128张图片的预测交叉熵。防止有的图片预测错了误差出现
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  global_steps=global_step%FLAGS.max_steps
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_steps,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)