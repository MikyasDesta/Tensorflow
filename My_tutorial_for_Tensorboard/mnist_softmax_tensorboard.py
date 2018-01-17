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

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
LOGDIR = "/tmp/mnist_tutorial_2/2"
LABELS = "/home/mike/tensorflow/tensorflow/examples/tutorials/mnist/labels_1024.tsv"
SPRITES = "/home/mike/tensorflow/tensorflow/examples/tutorials/mnist/sprite_1024.png"
FLAGS = None


def main(_):
  with tf.name_scope("input"):
   # Import data
   mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  with tf.name_scope("Fully_connected_layer"):
  # Create the model
   with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="Input")
   with tf.name_scope("Weights"):
    W = tf.Variable(tf.zeros([784, 10]), name="Weights")
   with tf.name_scope("biases"):
    b = tf.Variable(tf.zeros([10]), name="biases")

   with tf.name_scope("activation"):
    y = tf.matmul(x, W) + b

  # Define loss and optimizer
  with tf.name_scope("labels"):
   y_ = tf.placeholder(tf.float32, [None, 10],"labels")

  tf.summary.histogram("weights", W)
  tf.summary.histogram("biases", b)

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  with tf.name_scope("loss"):
   cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name="cross_entropy")
  tf.summary.scalar("loss", cross_entropy)
  with tf.name_scope("train"):
   train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, name="train_step")
   tf.add_to_collection('train_step', train_step)

  with tf.name_scope("accuracy"):
   correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

  tf.summary.scalar("Accuracy", accuracy)
  merged_summ = tf.summary.merge_all()

  #saver = tf.train.Saver()
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  writer = tf.summary.FileWriter(LOGDIR)
  writer.add_graph(sess.graph)

  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    summ = sess.run(merged_summ, feed_dict={x: batch_xs, y_: batch_ys})
    writer.add_summary(summ, _)
    #saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), _)

  # Test trained model

  print(sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
