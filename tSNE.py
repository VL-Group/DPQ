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
from DPQ import DPQ, IMAGE_WIDTH, IMAGE_HEIGHT
from Dataset import Dataset
import sklearn.manifold
from Utils import mAP

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
        vecs = Dataset('Cifar', 'database', 256, 256, 256)
        database, codebook = model.GetFeature(vecs.data)

        idx = np.random.permutation(database.label.shape[0])[:5000]
        Y = database.label[idx]
        codes = database.codes[idx]
        np.save('tSNE_Y', np.argmax(Y, axis=1))
        for i in range(4):
            X = mAP.Quantize_AQ(codes, codebook, i+1)
            print(X.shape, Y.shape)
            embedded = sklearn.manifold.TSNE(perplexity=50, init='pca', method='exact').fit_transform(X)
            np.save('tSNE_X_level_' + str(i+1), embedded)
            print(i, 'saved')
        embedded = sklearn.manifold.TSNE(perplexity=50, init='pca', method='exact').fit_transform(database.output[idx])
        np.save('tSNE_X_level_raw', embedded)
        print('raw saved')


if __name__ == '__main__':
    tf.app.run()
