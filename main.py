#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import print_function, division
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import os
from os import path
import numpy as np
import tensorflow as tf
from DPQ import DPQ

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("Dataset", "NUS", "The preferred dataset, \'CIFAR\', \'NUS\' or \'Imagenet\'")
tf.app.flags.DEFINE_string("Mode", "eval", "\'train\' or \'eval\'")
tf.app.flags.DEFINE_integer("BitLength", 32, "Binary code length")
tf.app.flags.DEFINE_integer("ClassNum", 21, "Label num of dataset")
tf.app.flags.DEFINE_integer("K", 256, "The centroids number of a codebook")
tf.app.flags.DEFINE_integer("PrintEvery", 50, "Print every ? iterations")
tf.app.flags.DEFINE_float("LearningRate", 1e-4, "Init learning rate")
tf.app.flags.DEFINE_integer("Epoch", 64, "Total epoches")
tf.app.flags.DEFINE_integer("BatchSize", 256, "Batch size")
tf.app.flags.DEFINE_string("Device", "0", "GPU device ID")
tf.app.flags.DEFINE_boolean("UseGPU", True, "Use /device:GPU or /cpu")
tf.app.flags.DEFINE_boolean("SaveModel", True, "Save model at every epoch done")
tf.app.flags.DEFINE_integer("R", 5000, "mAP@R, -1 for all")
tf.app.flags.DEFINE_float("Lambda", 0.1, "Lambda, decribed in paper")
tf.app.flags.DEFINE_float("Tau", 1, "Tau, decribed in paper")
tf.app.flags.DEFINE_float("Mu", 1, "Mu, decribed in paper")
tf.app.flags.DEFINE_float("Nu", 0.1, "Nu, decribed in paper")

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.Device)


def main(_):
    model = DPQ(FLAGS)
    a = "/device:GPU:0" if FLAGS.UseGPU else "/cpu:0"
    print("Using device:", a, "<-", FLAGS.Device)
    with tf.device(a):
        model.InitVariables()
        model.Train()
        # model.Save()


if __name__ == '__main__':
    tf.app.run()
