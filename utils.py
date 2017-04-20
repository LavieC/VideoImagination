"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import os
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def save_images(images, size, image_path):
  # images = (images+1.)/2.
  puzzle = merge(images, size)
  return scipy.misc.imsave(image_path, puzzle)

def merge(images, size):
  cdim = images.shape[-1]
  h, w = images.shape[1], images.shape[2]
  if cdim == 1 :
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j*h:j*h+h, i*w:i*w+w] = np.squeeze(image)
    return img
  else:
    img = np.zeros((h * size[0], w * size[1], cdim))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def SequenceToImageAndVideo(images,resize_height=320, resize_width=240):
  image_first = images[:,0,:,:,:]
  resized_images = tf.image.resize_images(image_first, [resize_width, resize_height])
  if resized_images.shape[-1] == 1:
    resized_images = tf.concat([resized_images,resized_images,resized_images],axis=-1)
  video = images
  return resized_images, video, image_first

def ReadInput(self,num_epochs=None, val=False,test=False):
  if val:
    filenames = tf.gfile.Glob(self.data_dir+'/surfing_val.tfrecords')
  elif test:
    filenames = tf.gfile.Glob(self.data_dir+'/surfing_test.tfrecords')
  else:
    filenames = tf.gfile.Glob(self.data_dir+'/surfing.tfrecords')
  filename_queue = tf.train.string_input_producer(filenames,num_epochs=num_epochs, shuffle=True)
  reader = tf.TFRecordReader()
  _, example = reader.read(filename_queue)
  feature_sepc = {
      self.features: tf.FixedLenSequenceFeature(
          shape=[self.image_width * self.image_width * self.c_dim], dtype=tf.float32)}
  _, features = tf.parse_single_sequence_example(
      example, sequence_features=feature_sepc)
  moving_objs = tf.reshape(
      features[self.features], [self.video_len, self.image_width, self.image_width, self.c_dim])
  examples = tf.train.shuffle_batch(
        [moving_objs],
        batch_size=self.batch_size,
        num_threads=self.batch_size,
        capacity=self.batch_size * 100,
        min_after_dequeue=self.batch_size * 4)
  return examples

def video_summary(name,Video,fs):
  shape =tf.shape(Video)
  sum=[]
  for x in range(fs):
    frame = tf.slice(Video,[0,x,0,0,0],[-1,1,-1,-1,-1])
    frame = tf.reshape(frame,[shape[0],shape[2],shape[3],shape[4]])
    sum.append(tf.summary.image(name, frame))
  return sum
