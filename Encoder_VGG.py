#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function, division
import time
import tensorflow as tf
from TripletLoss import batch_hard_triplet_loss, batch_all_triplet_loss
from Utils import *
from Dataset import Dataset
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]

VGG16_PATH = './data/models/vgg16.npy'
ALEX_PATH = './data/models/alexnet.npy'
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
NUS_WORD_DICT = './data/nus_txt/nuswide_wordvec.txt'
CIFAR_WORD_DICT = './data/cifar/cifar_wordvec.txt'

def convolve(i, k):
    return tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')


class Encoder_VGG(object):
    """docstring for Encoder."""

    def __init__(self, batchSize, classNum, stackLevel=4, subCenters=256, multiLabel=False, train=True):
        self.data_dict = np.load(VGG16_PATH, encoding='latin1').item()
        self._stackLevel = stackLevel
        self._subCenters = subCenters
        self._margin = 0.7
        self._train = train
        self.batch_size = batchSize
        self.n_class = classNum
        self._multiLabel = multiLabel
        # for primal test
        self.loss_type = 'cos_softmargin_multi_label'
        print("npy file loaded")
        print(self.loss_type)

    def Inference(self, input, labelHot):
        self.buildEncoder(input)
        self.ApplyLoss(labelHot)

    def vgg16(self, inp):
        """
        load variable from npy to build the VGG

        :param inp: rgb image [batch, height, width, 3] values scaled [0., 255.]
        """

        start_time = time.time()
        PrintWithTime(BarFormat("build model started (VGG-16)"))

        # input is images of [256, 256, 3], random crop and flip to [224, 224,
        # 3]
        distorted_image = tf.stack([tf.random_crop(tf.image.random_flip_left_right(each_image), [224, 224, 3]) for each_image in tf.unstack(inp)])

        self.train_layers = []
        self.train_last_layer = []
        self.classifyLastLayer = []

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=distorted_image)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        self.lrn1 = tf.nn.local_response_normalization(self.pool1,
                                                       depth_radius=radius,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       bias=bias)

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        self.lrn2 = tf.nn.local_response_normalization(self.pool2,
                                                       depth_radius=radius,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       bias=bias)

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.dropout(tf.nn.relu(self.fc6), 0.5) if self._train else tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.dropout(tf.nn.relu(self.fc7), 0.5) if self._train else tf.nn.relu(self.fc7)

        ''' ADD ONE MORE DENSE 4096 -> D '''
        # FC8
        # Output output_dim
        with tf.name_scope('fc8') as scope:
            fc8w = tf.Variable(tf.random_normal([4096, 300],
                                                dtype=tf.float32,
                                                stddev=1e-2), name='weights')
            fc8b = tf.Variable(tf.constant(0.0, shape=[300],
                                        dtype=tf.float32), name='biases')
            self.fc8l = tf.nn.bias_add(tf.matmul(self.relu7, fc8w), fc8b)
            self.fc8 = tf.nn.tanh(self.fc8l)
            self.train_last_layer += [fc8w, fc8b]
        # Classify
        # Output label_num
        with tf.name_scope('cls') as scope:
            clsw = tf.Variable(tf.random_normal([4096, self.n_class],
                                                dtype=tf.float32,
                                                stddev=1e-2), name='weights')
            clsb = tf.Variable(tf.constant(0.0, shape=[self.n_class],
                                        dtype=tf.float32), name='biases')
            self.cls = tf.nn.bias_add(tf.matmul(self.relu7, clsw), clsb)
            self.classifyLastLayer += [clsw, clsb]

        PrintWithTime(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.name_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME', name=name)

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)

            self.train_layers += [filt, conv_biases]

            return relu

    def fc_layer(self, bottom, name):
        with tf.name_scope(name) as scope:
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer.  Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            self.train_layers += [weights, biases]

            return fc

    def get_conv_filter(self, name):
        return tf.Variable(self.data_dict[name][0], name=name + "_filter")

    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name=name + "_biases")

    def get_fc_weight(self, name):
        return tf.Variable(self.data_dict[name][0], name=name + "_weights")

    def buildEncoder(self, inp):
        PrintWithTime("Deep Stacked Quantizer")
        PrintWithTime(BarFormat("Integrating NetPQ"))
        self.vgg16(inp)
        
        # output 300
        self.X = self.fc8
        with tf.variable_scope("Quantization") as _:
            self.HardCode = [None] * self._stackLevel

            D = self.X.get_shape().as_list()[-1]
            N = self.X.get_shape().as_list()[0]
            # [nlevel, cetners, D]
            self.Codebook = tf.get_variable("Codebook", shape=[self._stackLevel, self._subCenters, D])

            residual = self.X

            self.QSoft = tf.zeros([N, D])
            self.QHard = tf.zeros([N, D])

            self.SoftDistortion = tf.Variable(0.0, name="soft_distortion")
            self.HardDistortion = tf.Variable(0.0, name="hard_distortion")

            for level in range(self._stackLevel):
                codes = tf.gather(self.Codebook, level)
                # [N, d] · [d, k] -> [N, k]
                distance = tf.matmul(residual, tf.transpose(codes, [1, 0]))
                # [k]
                Cm_square = tf.reduce_sum(tf.square(codes), axis=1)
                # [N]
                Xm_square = tf.reduce_sum(tf.square(residual), axis=1)

                # meshgrid(k, N) -> [N, k]
                meshX, meshY = tf.meshgrid(Cm_square, Xm_square)

                # [N, k], l2 mod for all X and C
                mod = tf.sqrt(tf.multiply(meshX, meshY))

                # [N, k] distances, larger distance means more similar
                distance = distance / mod

                soft = tf.matmul(tf.nn.softmax(distance), codes)
                code = tf.argmax(distance, axis=1)
                self.HardCode[level] = code
                hard = tf.gather(codes, code)

                residual -= hard

                self.QSoft += soft
                self.QHard += hard


            self.SoftDistortion += tf.reduce_mean(
                tf.square(self.X - self.QSoft))
            self.HardDistortion += tf.reduce_mean(
                tf.square(self.X - self.QHard))


            PrintWithTime("NetPQ output: ")
            print("Qsoft:", self.QSoft.get_shape())
            print("Qhard:", self.QHard.get_shape())    


    def ApplyLoss(self, label):
        label = tf.cast(label, tf.float32)
        # loss function
        if self.loss_type == 'cos_margin_multi_label':
            # apply L = sum(sum(max(0, delta - cos1 + cos2)))
            # equation (1) in paper
            word_dict = tf.constant(np.loadtxt(NUS_WORD_DICT if self.n_class>10 else CIFAR_WORD_DICT), dtype=tf.float32)

            # hard margin just set delta = constant
            margin_param = tf.constant(self._margin, dtype=tf.float32)

            # N: batch_num, L: label_dim, D: 300
            # img_label: N * L
            # word_dic: L * D
            # v_label: N * L * D
            # the correct label embedding {Vi}
            v_label = tf.multiply(tf.expand_dims(label, 2), tf.expand_dims(word_dict, 0))

            # img_last: N * D
            # ip_1: N * L
            # dot product: {Vi}.  T * Z, here has broadcasting
            ip_1 = tf.reduce_sum(tf.multiply(tf.expand_dims(self.X, 1), v_label), 2)

            # mod_1: N * L
            # || Vi || * || Z ||
            v_label_mod = tf.multiply(tf.expand_dims(tf.ones([self.batch_size, self.n_class]), 2), tf.expand_dims(word_dict, 0))
            mod_1 = tf.sqrt(tf.multiply(tf.expand_dims(tf.reduce_sum(tf.square(self.X), 1), 1), tf.reduce_sum(tf.square(v_label_mod), 2)))

            # mod_1 = tf.where(tf.less(mod_1_1, tf.constant(0.0000001)),
            # tf.ones([self.batch_size, self.n_class]), mod_1_1)
            # cos_1: N * L
            cos_1 = tf.div(ip_1, mod_1)

            # all label embedding {V}
            ip_2 = tf.matmul(self.X, word_dict, transpose_b=True)

            # multiply ids to inner product
            # ip_2 = tf.multiply(ip_2_1, ids_dict)

            def reduce_shaper(t):
                return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])

            # same calculation as mod_1, 很迷的操作
            mod_2_2 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(self.X)), reduce_shaper(tf.square(word_dict)), transpose_b=True))
            # pick where label is 1, set them to 0, 相当于去掉对的词
            mod_2 = tf.where(tf.less(mod_2_2, tf.constant(0.0000001)), tf.ones([self.batch_size, self.n_class]), mod_2_2)
            # cos_2: N * L
            cos_2 = tf.div(ip_2, mod_2)

            # cos - cos: N * L * L
            # delta - cos1 + cos2
            cos_cos_1 = tf.subtract(margin_param, tf.subtract(tf.expand_dims(cos_1, 2), tf.expand_dims(cos_2, 1)))
            # we need to let the wrong place be 0
            # only use i∈Y, as in the first sum
            cos_cos = tf.multiply(cos_cos_1, tf.expand_dims(label, 2))
            # sum up
            cos_loss = tf.reduce_sum(tf.maximum(tf.constant(0, dtype=tf.float32), cos_cos))
            # average them here is the total num of sample
            self.cos_loss = tf.div(cos_loss, tf.multiply(tf.constant(self.n_class, dtype=tf.float32), tf.reduce_sum(label)))

        elif self.loss_type == 'cos_softmargin_multi_label':
            word_dict = tf.constant(np.loadtxt(NUS_WORD_DICT if self.n_class>10 else CIFAR_WORD_DICT), dtype=tf.float32)
            # margin_param = tf.constant(self._margin, dtype=tf.float32)

            # N: batchsize, L: label_dim, D: 300
            # img_label: N * L
            # word_dic: L * D
            # v_label: N * L * D
            v_label = tf.multiply(tf.expand_dims(label, 2), tf.expand_dims(word_dict, 0))
            # img_last: N * D
            # ip_1: N * L
            ip_1 = tf.reduce_sum(tf.multiply(tf.expand_dims(self.X, 1), v_label), 2)
            # mod_1: N * L
            v_label_mod = tf.multiply(tf.expand_dims(tf.ones([self.batch_size, self.n_class]), 2), tf.expand_dims(word_dict, 0))
            mod_1 = tf.sqrt(tf.multiply(tf.expand_dims(tf.reduce_sum(tf.square(self.X), 1), 1), tf.reduce_sum(tf.square(v_label_mod), 2)))
            # mod_1 = tf.where(tf.less(mod_1_1, tf.constant(0.0000001)),
            # tf.ones([self.batch_size, self.n_class]), mod_1_1)
            # cos_1: N * L
            cos_1 = tf.div(ip_1, mod_1)

            ip_2 = tf.matmul(self.X, word_dict, transpose_b=True)

            # multiply ids to inner product
            # ip_2 = tf.multiply(ip_2_1, ids_dict)

            def reduce_shaper(t):
                return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])

            mod_2_2 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(self.X)), reduce_shaper(tf.square(word_dict)), transpose_b=True))
            mod_2 = tf.where(tf.less(mod_2_2, tf.constant(0.0000001)), tf.ones([self.batch_size, self.n_class]), mod_2_2)
            # cos_2: N * L
            cos_2 = tf.div(ip_2, mod_2)

            # word_dic: L * D
            # ip_3: L * L
            # compute soft margin
            ip_3 = tf.matmul(word_dict, word_dict, transpose_b=True)
            # use word_dic to avoid 0 in /
            mod_3 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(word_dict)), reduce_shaper(tf.square(word_dict)), transpose_b=True))

            # soft_margin is explained as paper
            margin_param = 1 - (ip_3 / mod_3)

            # cos - cos: N * L * L
            cos_cos_1 = tf.subtract(tf.expand_dims(margin_param, 0), tf.subtract(tf.expand_dims(cos_1, 2), tf.expand_dims(cos_2, 1)))
            # we need to let the wrong place be 0
            cos_cos = tf.multiply(cos_cos_1, tf.expand_dims(label, 2))

            cos_loss = tf.reduce_sum(tf.maximum(tf.constant(0, dtype=tf.float32), cos_cos))
            self.cos_loss = tf.div(cos_loss, tf.multiply(tf.constant(self.n_class, dtype=tf.float32), tf.reduce_sum(label)))

        self.classify = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls, labels=label)
                                       if self._multiLabel == False else tf.nn.sigmoid_cross_entropy_with_logits(logits=self.cls, labels=label))
        print("Multi Label:", self._multiLabel)
        
        self.loss = self.cos_loss + self.classify
        """ Quantization Loss """
        # JCL
        self.JointCenter = tf.reduce_mean(tf.square(self.QSoft-self.QHard))
        PrintWithTime(BarFormat("Loss built"))