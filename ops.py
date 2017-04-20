import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

def conv2d_apply(stimage, kernels, self):

  kernel_times = tf.split(kernels, self.output_frames, 1) 
  stimage_batchs = tf.split(stimage, self.batch_size, 0)

  frames = []
  for kernel_t in kernel_times:
    kernel_t = tf.reshape(kernel_t,
        [self.batch_size, self.conv_size, self.conv_size, 1, self.sequence_len])
    kernel_t = tf.tile(kernel_t,[1,1,1,self.c_dim])
    kernel_t = tf.nn.relu(kernel_t - 1e-12) + 1e-12
    kernel_t /= tf.reduce_sum(kernel_t, [1, 2, 3], keep_dims=True)
    kernel_batchs = tf.split(kernel_t, self.batch_size, 0)

    transformed = []
    for kernel, preimg in zip(kernel_batchs, stimage_batchs):
      kernel = tf.squeeze(kernel, axis = 0)
      transformed.append(
          tf.nn.depthwise_conv2d(preimg, kernel, [1, 1, 1, 1], 'SAME'))
    transformed = tf.concat(transformed, axis=0)
    transformed = tf.split(transformed, self.sequence_len, axis=3)
    transformed = tf.stack(transformed, axis=1)
    frames.append(transformed)
  return frames

def affine_apply(stimage, kernels, self):
  identity_params = tf.convert_to_tensor(
      np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32))

  kernels = tf.nn.tanh(kernels)*0.1
  kernels = tf.reshape(kernels,
        [self.batch_size, self.output_frames, 6, self.sequence_len])
  kernel_times = tf.split(kernels, self.output_frames, axis=1)

  frames = []
  for kernel_t in kernel_times:
    kernel_nums = tf.split(kernel_t, self.sequence_len, axis=3)
    transformed = []

    for kernel in kernel_nums:
      params = tf.squeeze(kernel, axis=[1,3])
      params += identity_params
      transformed.append(transformer(stimage, params))
    transformed = tf.stack(transformed, axis=1)
    frames.append(transformed)
  return frames


def volumetric_apply(layered_image, stimage, raw_trans_1, raw_trans_2,self):
  '''
  layed_image: list of batch *num* h *w *c
  raw_trans_1: batch * -1
  raw_trans_2: batch * -1
  '''
  trans_a = tf.reshape(raw_trans_1,[self.batch_size,self.sequence_len,5,5,1,1])
  trans_a = tf.nn.relu(trans_a- (1e-12)) + (1e-12)
  norm_factor = tf.reduce_sum(trans_a,[1,2,3],keep_dims=True)
  trans_a /= norm_factor

  trans_b = tf.reshape(
          tf.nn.softmax(tf.reshape(raw_trans_2, [-1, self.sequence_len])),
          [int(self.batch_size), self.input_width, self.input_height, self.sequence_len])
  mask_list = tf.split(trans_b, self.sequence_len, axis = 3)

  ## with conv3d
  frames_1 = []
  for image_t in layered_image:
    image_channels = tf.split(image_t, self.c_dim, axis = 4)
    d3ed = []
    for image_c in image_channels:
      ted_batchs = tf.split(image_c,self.batch_size,axis=0)
      tra_batchs = tf.split(trans_a,self.batch_size,axis=0)
      sketch = []
      for kernel, preimg in zip(tra_batchs, ted_batchs):
        kernel = tf.squeeze(kernel, axis = 0)
        sketch.append(
          tf.nn.conv3d(preimg, kernel, [1, 1, 1, 1, 1], 'SAME'))
      d3ed.append(tf.concat(sketch,0))
    image_d3ed = tf.concat(d3ed, axis=4)

    image_filters_1 = tf.split(image_d3ed, self.sequence_len, axis=1)

    output_1 = tf.zeors(stimage.get_shape())
    for layer_1, layer_2, mask in zip(image_filters_1, image_filters_2, mask_list[1:]):
      output_1 += tf.squeeze(layer_1,axis=1) * mask

    frames_1.append(output_1)

  vid_1 = tf.stack(frames_1, axis=1)

  return vid_1, vid_2 #vid_1 con3d

def transformer(U, theta, out_size=[240,320], name='SpatialTransformer', **kwargs):
    """
    From https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            num_channels = tf.shape(input_dim)[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            return output

    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output


def batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer'):

    with tf.variable_scope(name):
        num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
        indices = [[i]*num_transforms for i in xrange(num_batch)]
        input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
        return transformer(input_repeated, thetas, out_size)


def conv3d(input_, output_dim,
       k_h=4, k_w=4, k_d=2, d_h=2, d_w=2, d_d=2, stddev=0.02,
       name="conv3d",with_w=False):

  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w,1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    if with_w:
      return conv, w, biases
    else:
      return conv

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias

def conv2d(input_, output_dim,
       k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
       name="conv2d",with_w=False,padding='SAME'):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    if with_w:
      return conv, w, biases
    else:
      return conv
