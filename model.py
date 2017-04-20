from __future__ import division
import os
import time
import numpy as np
import tensorflow as tf

from ops import *
from utils import *
from glob import glob
from caffe_classes import class_names

class Vid_Imagine(object):
  def __init__(self, sess,
         batch_size=64,
         num_epochs = 25,
         image_height=64, image_width=64, c_dim=1,

         conv_size = 9,
         sequence_len = 5,
         trans_par = 6,
         transformation='affine_transformation',

         dataset_name='MNIST',
         data_dir = '../DataBase/data',
         feature = 'digits',

         z_dim=100,
         emcode_len = 512,
         clamp_lower=-0.01,clamp_upper=0.01,
         output_frames=4,
         video_len=5,
         is_flatten=False,
         is_conv=True):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      z_dim: Dimension of dim for Z. [100]
      emcode_len: Dimension of encoded condition code [512]
      clamp_lower: clamp parameters in WGAN [-0.01]

      conv_size: convolution kernel size [9,16]
      sequence_len: transformation sequence length [5,10]
      trans_par: number of parameters in transformation [6,9*9,16*16]
      transformation: transformation model type [affine_transformation,conv_transformation]

      output_frames: number of frames reconstructed
      video_len: number of frames in imaginary video
      is_flatten: Flatten image as condition code 
      is_conv: Finetune alexnet or use custom conv

    """
    self.sess = sess
    # batch info
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    # input info
    self.image_height = image_height
    self.image_width = image_width
    self.c_dim = c_dim
    # dataset info
    self.dataset_name = dataset_name
    self.input_pattern = '/'+dataset_name+'.tfrecords'
    self.video_len = video_len
    self.data_dir = data_dir
    self.feature = feature
    # output info
    self.conv_size = conv_size
    self.trans_par = trans_par
    self.sequence_len = sequence_len
    self.output_frames = output_frames
    # parameter info
    self.is_flatten = is_flatten
    self.is_conv = is_conv
    self.transformation = transformation
    self.z_dim = z_dim
    self.emcode_len = emcode_len
    self.clamp_lower = clamp_lower
    self.clamp_upper = clamp_upper

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.e_bn1 = batch_norm(name='e_bn1')
    self.e_bn2 = batch_norm(name='e_bn2')
    self.e_bn3 = batch_norm(name='e_bn3')
    self.e_bn4 = batch_norm(name='e_bn4')
    self.e_bn5 = batch_norm(name='e_bn5')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    self.build_model()

  def build_model(self):
    ###Load Alex Net###
    net_data = np.load(open(self.data_dir+"/bvlc_alexnet.npy","rb"),encoding="latin1").item()
    
    ###Sample Noise###
    z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
    self.z = z

    ###Train data flow###
    images = ReadInput(self, num_epochs = self.num_epochs)
    stimage, real_video, stimage_64 = SequenceToImageAndVideo(images)
    self.imaginary = self.generator(z, stimage, stimage_64, net_data)
    
    ###Validate data flow###
    val_images = ReadInput(self, val=True)
    val_stimage, val_real_video, val_stimage_64 = SequenceToImageAndVideo(val_images)
    self.samplers = self.generator(z, val_stimage, val_stimage_64)

    ###Loss function###
    true_logit = self.VideoCritic(real_video)
    fake_logit = self.VideoCritic(self.imaginary,reuse = True)
    self.d_loss = -tf.reduce_mean(fake_logit - true_logit)
    self.g_loss = -tf.reduce_mean(-fake_logit)

    ###TensorBoard visualization###
    self.z_sum = tf.summary.histogram("z", z)
    self.true_sum = tf.summary.histogram("d", true_logit)
    self.fake_sum = tf.summary.histogram("d_", fake_logit)
    self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
    self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
    self.imaginary_sum = video_summary("imaginary", self.imaginary,self.output_frames+1)

    ###Variable preparing###
    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    self.d_clamp_op = [tf.assign(var, tf.clip_by_value(var, self.clamp_lower, self.clamp_upper)) for var in self.d_vars]
    
    self.saver = tf.train.Saver()

  def train(self, config):
    ################
    # optimization #
    ################
    d_optim = tf.train.RMSPropOptimizer(config.learning_rate) \
               .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.RMSPropOptimizer(config.learning_rate) \
               .minimize(self.g_loss, var_list=self.g_vars)
    ##################
    # initialization #
    ##################
    self.coord = tf.train.Coordinator()
    EPOCH = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)[0].name
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    ########
    # log  #
    ########
    self.g_sum = tf.summary.merge([self.z_sum, self.fake_sum, self.imaginary_sum, self.g_loss_sum])
    self.d_sum = tf.summary.merge([self.z_sum, self.true_sum, self.d_loss_sum])
    self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

    ##################
    # validation set #
    ##################
    sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))
    sample_multi_z = [np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim)) for i in range(5)]

    ###################
    # load checkpoint #
    ###################
    could_load, checkpoint_counter = self.load(config.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    counter = 1
    start_time = time.time()
    grid_size = int(sqrt(self.batch_size))
    tf.get_default_graph().finalize()
    ###############
    # Start epoch #
    ###############
    try:
      while not self.coord.should_stop():
          batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

          if counter < 10 or counter%500 == 0:
            Diter = 10
          else:
            Diter = config.Diter

          ####################
          # Update Critic network #
          ####################
          print("====Update Critic====") 
          for j in range(Diter):
            if counter % 100 ==99 and j == 0:
              _, summary_str,_ = self.sess.run([d_optim, self.d_sum,self.d_clamp_op], 
                                             feed_dict={self.z: batch_z})
              self.writer.add_summary(summary_str, counter) 
            else:
              _,_= self.sess.run([d_optim,self.d_clamp_op], feed_dict={self.z: batch_z})

          ####################
          # Update G network #
          ####################
          print("====Update Generator====")
          if counter % 100 ==99:
            _, summary_str, errD, errG, epoch = self.sess.run([g_optim, self.g_sum, self.d_loss,self.g_loss,EPOCH],
                                              feed_dict={ self.z: batch_z})
            self.writer.add_summary(summary_str, counter)
          else:
             _, errD, errG, epoch = self.sess.run([g_optim, self.d_loss,self.g_loss, EPOCH],
                                              feed_dict={ self.z: batch_z})

          ###########
          # Monitor #
          ###########
          counter += 1
          print("Epoch: [%2d/%2d] Counter: [%2d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
    	           % (epoch, self.num_epochs, counter, time.time() - start_time, errD, errG))

          ##############
          # validation #
          ##############
          if np.mod(counter, 5*config.validation) == 1:
            print("~!~!~!~!Multiple Sampling Validation~!~!~!~")
            for i in range(len(sample_multi_z)):
              samples = self.sess.run([self.samplers],feed_dict={self.z: sample_multi_z[i] })
              for times in range(self.output_frames+1):
                i_sample = samples[:,times,:,:,:]
                save_images(i_sample, [grid_size,grid_size],
                            './samples/train_{:02d}_{:02d}_{:02d}.png'.format(counter, i,times))

          elif np.mod(counter, config.validation) == 1:
            print("~!~!~!~!Single Sampling Validation~!~!~!~")
            samples = self.sess.run([self.samplers], feed_dict={self.z: sample_z })
            for times in range(self.output_frames+1):
              i_sample = samples[:,times,:,:,:]
              save_images(i_sample, [grid_size,grid_size],
                            './samples/train_{:02d}_{:02d}.png'.format(counter, times))

          ##############
          # save model #
          ##############
          if np.mod(counter, config.save_times) == 2:
    	      self.save(config.checkpoint_dir, counter)

    except tf.errors.OutOfRangeError:
      print 'Done training -- epoch limit reached'
    finally:
      self.coord.request_stop()

    ### Wait for threads to finish.###
    self.coord.join(self.threads)
    self.sess.close()

  def VideoCritic(self, video, reuse=False):
    with tf.variable_scope("VideoCritic") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv3d(video, 64, k_d=4, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv3d(h0, 64*2, k_d=4, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv3d(h1, 64*4, k_d=4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv3d(h2, 64*8, name='d_h3_conv')))
      logits = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

      return logits

  def generator(self, z, stimage, stimage_64, net_data):
    with tf.variable_scope("generator") as scope:
      if self.is_flatten:
        flat = tf.reshape(stimage,[self.batch_size,-1])
        f7_fine ,self.f7f_w, self.f7f_b = linear(flat,self.emcode_len,'g_e1_lin',with_w=True)
        e4 = tf.nn.relu(f7_fine)
        emb = tf.concat([e4, z],1)
      elif self.is_conv:
        e0 = lrelu(conv2d(stimage_64, 128, name='g_e0_conv'))
        e1 = lrelu(self.e_bn1(conv2d(e0, 256, name='g_e1_conv')))
        e2 = lrelu(self.e_bn2(conv2d(e1, 512, name='g_e2_conv')))
        e3 = lrelu(self.e_bn3(conv2d(e2, 1024, name='g_e3_conv')))
        e3 = tf.reshape(e3,[self.batch_size,-1])
        e4_1 = linear(e3,self.emcode_len*2,'g_e4_1_lin')
        e4_1 = tf.nn.relu(self.e_bn4(e4_1))
        e4_2 = linear(e4_1,self.emcode_len,'g_e4_2_lin')
        e4_2 = tf.nn.relu(self.e_bn5(e4_2))
        emb = tf.concat([e4_2, z],1)
      else:
        with tf.variable_scope("alex"):
          #conv1
          #conv(9, 9, 96, 4, 4, padding='VALID', name='conv1')
          k_h = 9; k_w = 9; c_o = 96; s_h = 4; s_w = 4
          conv1W = tf.get_variable("g_e_conv1w", [9, 9, stimage_64.get_shape()[-1], 96],
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
          conv1b = tf.get_variable("g_e_conv1b", [96], initializer=tf.constant_initializer(0.0))
          conv1_in = conv(stimage_64, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
          conv1 = tf.nn.relu(conv1_in)

          #lrn1
          #lrn(2, 2e-05, 0.75, name='norm1')
          radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
          lrn1 = tf.nn.local_response_normalization(conv1,  depth_radius=radius,
                                                            alpha=alpha,
                                                            beta=beta,
                                                            bias=bias)

          #maxpool1
          #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
          k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
          maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


          #conv2
          #conv(5, 5, 256, 1, 1, group=2, name='conv2')
          k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
          conv2W = tf.get_variable(name = "g_e_conv2w",initializer = net_data["conv2"][0])
          conv2b = tf.get_variable(name = "g_e_conv2b",initializer = net_data["conv2"][1])
          conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
          conv2 = tf.nn.relu(conv2_in)


          #lrn2
          #lrn(2, 2e-05, 0.75, name='norm2')
          radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
          lrn2 = tf.nn.local_response_normalization(conv2,  depth_radius=radius,
                                                            alpha=alpha,
                                                            beta=beta,
                                                            bias=bias)

          #maxpool2
          #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
          k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
          maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

          #conv3
          #conv(3, 3, 384, 1, 1, name='conv3')
          k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
          conv3W = tf.get_variable(name = "g_e_conv3w",initializer = net_data["conv3"][0])
          conv3b = tf.get_variable(name = "g_e_conv3b",initializer = net_data["conv3"][1])
          conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
          conv3 = tf.nn.relu(conv3_in)

          #conv4
          #conv(3, 3, 384, 1, 1, group=2, name='conv4')
          k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
          conv4W = tf.get_variable(name = "g_e_conv4w",initializer = net_data["conv4"][0])
          conv4b = tf.get_variable(name = "g_e_conv4b",initializer = net_data["conv4"][1])
          conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
          conv4 = tf.nn.relu(conv4_in)


          #conv5
          #conv(3, 3, 256, 1, 1, group=2, name='conv5')
          k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
          conv5W = tf.get_variable(name = "g_e_conv5w",initializer = net_data["conv5"][0])
          conv5b = tf.get_variable(name = "g_e_conv5b",initializer = net_data["conv5"][1])
          conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
          conv5 = tf.nn.relu(conv5_in)

          #maxpool5
          #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
          k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
          maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

          #fc6
          #fc(4096, name='fc6')
          fc6W = tf.get_variable(name = "g_e_fc6w",initializer = net_data["fc6"][0])
          fc6b = tf.get_variable(name = "g_e_fc6b",initializer = net_data["fc6"][1])
          fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [self.batch_size, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

          fc7
          fc(4096, name='fc7')
          fc7W = tf.get_variable(name = "g_e_fc7w",initializer = net_data["fc7"][0])
          fc7b = tf.get_variable(name = "g_e_fc7b",initializer = net_data["fc7"][1])
          fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

          f7_fine ,self.f7f_w, self.f7f_b = linear(fc7,self.emcode_len,'g_e1_lin',with_w=True)
          e4 = tf.nn.relu(f7_fine)
          emb = tf.concat([e4, z],1)

      ### transformation genrator fc1
      h0 = linear(emb, emb.get_shape()[1]*2, 'g_h0_lin')
      h0 = tf.nn.relu(self.g_bn0(h0))

      ### transformation genrator fc2
      h1 = linear(h0, emb.get_shape()[1], 'g_h1_lin')
      h1 = tf.nn.relu(self.g_bn1(h1))

      kernel_2d_len = self.trans_par*self.sequence_len*self.output_frames
      kernel_3d_len_1 = self.sequence_len*5*5
      kernel_3d_len_2 = self.sequence_len*self.input_height*self.input_width

      ### transformation genrator fc2
      h2 = linear(h1, kernel_2d_len+kernel_3d_len_1+kernel_3d_len_2, 'g_h2_lin')
      kernel_2d = tf.slice(h2,[0,0],[-1,kernel_2d_len])
      kernel_3d_1 = tf.slice(h2,[0,kernel_2d_len],[-1,kernel_3d_len_1])
      kernel_3d_2 = tf.slice(h2,[0,kernel_3d_len_1+kernel_2d_len],[-1,-1])

      ### transformation applying
      if self.transformation == 'affine_transformation':
        self.transformed = affine_apply(stimage_64, kernel_2d, self)
      elif self.transformation == 'conv_transformation':
        self.transformed = conv2d_apply(stimage_64, kernel_2d, self)

      ### Volumetric merge network ###
      frames_1,frames_2 = volumetric_apply(self.transformed, stimage_64, kernel_3d_1, kernel_3d_2, self)

      firstframe = tf.expand_dims(stimage_64, axis=1)
      video =  [firstframe,frames_1]
      return tf.concat(video,1)

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.image_height, self.image_width)

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
