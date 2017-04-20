import os
import scipy.misc
import numpy as np

from model import Vid_Imagine
from test import Vid_Imagine_Test
from utils import pp, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 5e-5, "Learning rate of for adam [0.0002]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 64, "The size of image to use [64,128]")
flags.DEFINE_integer("input_width", 64, "The size of image to use [64, 128]")
flags.DEFINE_integer("train_size", 64000, "The size of train images [np.inf]")
flags.DEFINE_integer("Diter", 5, "time of iteration of D")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [1,3]")
flags.DEFINE_integer("validation", 50, "validate frequency")
flags.DEFINE_integer("save_times", 200, "checkpoint save frequency")

flags.DEFINE_string("dataset_dir", "../DataBase/data", "The directory of dataset")
flags.DEFINE_string("dataset", "d2shape", "The name of dataset [MNIST, UCF101, d2shape]")
flags.DEFINE_string("feature", "moving_objs", "The feature name of the tfrecords [digits, frames, moving_objs]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "validation", "Directory name to save the validat  samples [validation]")
flags.DEFINE_string("test_dir", "test", "Directory name to save the test samples [samples]")

flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  if not os.path.exists(FLAGS.test_dir):
    os.makedirs(FLAGS.test_dir)


  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  if FLAGS.is_train:
    with tf.Session(config=run_config) as sess:
      v_imagie = Vid_Imagine(
          sess,
          image_width=FLAGS.input_width,
          image_height=FLAGS.input_height,
          num_epochs=FLAGS.epoch,
          batch_size=FLAGS.batch_size,
          c_dim=FLAGS.c_dim,
          feature=FLAGS.feature, 
          data_dir=FLAGS.dataset_dir,
          dataset_name=FLAGS.dataset)
      show_all_variables()
      v_imagie.train(FLAGS)
  else:
    with tf.Session(config=run_config) as sess:
      v_imagie = Vid_Imagine_Test(
          sess,
          image_width=FLAGS.input_width,
          image_height=FLAGS.input_height,
          num_epochs = FLAGS.epoch,
          batch_size=FLAGS.batch_size,
          c_dim=FLAGS.c_dim,
          feature=FLAGS.feature, 
          data_dir=FLAGS.dataset_dir,
          dataset_name=FLAGS.dataset)
      show_all_variables()
      v_imagie.test(FLAGS)

if __name__ == '__main__':
  tf.app.run()
