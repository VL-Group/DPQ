#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function, division
import time
import tensorflow as tf
from TripletLoss import *
from Utils import *
from Dataset import Dataset
from functools import reduce
import math

ALEX_PATH = './data/models/alexnet.npy'
NUS_WORD_DICT = './data/nus21/nus21_wordvec.txt'
IMG_WORD_DICT = './data/imagenet/imagenet_wordvec.txt'
CIFAR_WORD_DICT = './data/cifar/cifar_wordvec.txt'
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


def LMP(tensor, p=2):
    # num, row, col, filter
    n, r, c, f = tensor.get_shape().as_list()
    # note the overlap
    length = int(math.ceil(r / float(p)))
    actual = r / float(p)
    print(length, actual)
    slices = list()
    # Split the tensor horizontally
    # [13, 13, 256] -> p * [ ceil(13/p), 13, 256]
    j = 0
    k = 0.0
    for _ in range(p):
        j += length
        k += actual
        if j - k >= 1:
            j -= 1
        i = j - length
        slices.append(tf.slice(tensor, [0, i, 0, 0], [n, length, c, f]))
        # print("[0:%d, %d:%d, 0:%d, 0:%d]"%(n, i, (i+length), c, f))
        # print("actual shape: ", slices[-1].get_shape().as_list())
    out = list()
    # for each split, do max pooling
    for s in slices:
        # [13//p, 13, 256] -> [ {(13 // p - 3) // 2 + 1} , 6, 256]
        out.append(tf.nn.max_pool(s, ksize=[1, 3, 3, 1], strides=[
                   1, 2, 2, 1], padding='VALID'))
    return out


def convolve(i, k):
    return tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')


class Encoder_Alex(object):
    """In Encoder(DVSQ), output feature dim is constant: 300 (The word embedding dim)"""
    def __init__(self, batchSize, class_num, Lambda, subLevel=4, subCenters=256, multiLabel=False, train=True):
        self._stackLevel = subLevel
        self._subCenters = subCenters
        self._margin = 0.7
        self._train = train
        self.batch_size = batchSize
        self.n_class = class_num
        self._multiLabel = multiLabel
        self._lambda = Lambda
        # for primal test
        self.loss_type = 'cos_softmargin_multi_label'
        print("npy file loaded")
        print(self.loss_type)

    def Inference(self, input, labelHot):
        self.buildEncoder(input)

        self.ApplyLoss(labelHot)

    def alexnet(self, inp):
        self.deep_param_img = {}
        self.train_layers = []
        self.train_last_layer = []
        self.classifyLastLayer = []

        start_time = time.time()
        PrintWithTime(BarFormat("build model started (AlexNet)"))
        net_data = np.load(ALEX_PATH, encoding="latin1").item()

        # swap(2,1,0)
        reshaped_image = tf.cast(inp, tf.float32)
        tm = tf.Variable([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=tf.float32)
        reshaped_image = tf.reshape(
            reshaped_image, [self.batch_size * IMAGE_WIDTH * IMAGE_HEIGHT, 3])
        reshaped_image = tf.matmul(reshaped_image, tm)
        reshaped_image = tf.reshape(
            reshaped_image, [self.batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, 3])

        IMAGE_SIZE = 227
        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # Randomly crop a [height, width] section of each image
        distorted_image = tf.stack([tf.random_crop(tf.image.random_flip_left_right(
            each_image), [height, width, 3]) for each_image in tf.unstack(reshaped_image)])

        # Zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[
                               1, 1, 1, 3], name='img-mean')
            distorted_image = distorted_image - mean

        ''' ########### FOLLOWING STRUCTURES IN ALEXNET ########### '''

        # Conv1
        # Output 96, kernel 11, stride 4
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(net_data['conv1'][0], name='weights')
            conv = tf.nn.conv2d(distorted_image, kernel, [
                                1, 4, 4, 1], padding='VALID')
            biases = tf.Variable(net_data['conv1'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv1'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        # Pool1
        self.pool1 = tf.nn.max_pool(self.conv1,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool1')

        # LRN1
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0

        ''' FOLLOWING LRN IN ALEXNET '''
        self.lrn1 = tf.nn.local_response_normalization(self.pool1,
                                                       depth_radius=radius,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       bias=bias)

        # Conv2
        # Output 256, pad 2, kernel 5, group 2
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(net_data['conv2'][0], name='weights')
            group = 2

            input_groups = tf.split(
                self.lrn1, axis=3, num_or_size_splits=group)
            kernel_groups = tf.split(kernel, axis=3, num_or_size_splits=group)
            output_groups = [convolve(i, k)
                             for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(output_groups, 3)

            biases = tf.Variable(net_data['conv2'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv2'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        # Pool2
        self.pool2 = tf.nn.max_pool(self.conv2,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool2')

        # LRN2
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        self.lrn2 = tf.nn.local_response_normalization(self.pool2,
                                                       depth_radius=radius,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       bias=bias)

        # Conv3
        # Output 384, pad 1, kernel 3
        with tf.name_scope('conv3') as scope:
            kernel = tf.Variable(net_data['conv3'][0], name='weights')
            conv = tf.nn.conv2d(self.lrn2, kernel, [
                                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv3'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv3'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        # Conv4
        # Output 384, pad 1, kernel 3, group 2
        with tf.name_scope('conv4') as scope:
            kernel = tf.Variable(net_data['conv4'][0], name='weights')
            group = 2

            input_groups = tf.split(
                self.conv3, axis=3, num_or_size_splits=group)
            kernel_groups = tf.split(kernel, axis=3, num_or_size_splits=group)
            output_groups = [convolve(i, k)
                             for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(output_groups, 3)
            biases = tf.Variable(net_data['conv4'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv4'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        # Conv5
        # Output 256, pad 1, kernel 3, group 2
        with tf.name_scope('conv5') as scope:
            kernel = tf.Variable(net_data['conv5'][0], name='weights')
            group = 2

            input_groups = tf.split(
                self.conv4, axis=3, num_or_size_splits=group)
            kernel_groups = tf.split(kernel, axis=3, num_or_size_splits=group)
            output_groups = [convolve(i, k)
                             for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(output_groups, 3)
            biases = tf.Variable(net_data['conv5'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv5'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        # Pool5
        # [13, 13, 256] -> [6, 6, 256]

        # Local Max Pooling
        # out = LMP(self.conv5, p=3)
        # combine the pooled results on same axis
        # p * [ {(13 // p - 3) // 2 + 1} , 6, 256] -> [ p * {(13 // p - 3) // 2 + 1} , 6, 256]
        # self.pool5 = tf.concat(out, axis=1, name='pool5')
        # print(self.pool5.get_shape().as_list())

        self.pool5 = tf.nn.max_pool(self.conv5,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool5')

        # FC6
        # Output 4096
        with tf.name_scope('fc6') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc6w = tf.Variable(net_data['fc6'][0], name='weights')
            fc6b = tf.Variable(net_data['fc6'][1], name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            self.fc5 = pool5_flat
            fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
            self.fc6 = tf.nn.dropout(tf.nn.relu(
                fc6l), 0.5) if self._train else tf.nn.relu(fc6l)
            self.deep_param_img['fc6'] = [fc6w, fc6b]
            self.train_layers += [fc6w, fc6b]

        # FC7
        # Output 4096
        with tf.name_scope('fc7') as scope:
            fc7w = tf.Variable(net_data['fc7'][0], name='weights')
            fc7b = tf.Variable(net_data['fc7'][1], name='biases')
            fc7l = tf.nn.bias_add(tf.matmul(self.fc6, fc7w), fc7b)
            self.fc7 = tf.nn.dropout(tf.nn.relu(
                fc7l), 0.5) if self._train else tf.nn.relu(fc7l)
            self.deep_param_img['fc7'] = [fc7w, fc7b]
            self.train_layers += [fc7w, fc7b]

        # ''' ADD ONE MORE DENSE 4096 -> D '''
        # FC8
        # Output output_dim

        
        # with tf.name_scope('attetion') as scope:
        #     w_k = tf.Variable(tf.random_normal([4096, 2048],
        #                                         dtype=tf.float32,
        #                                         stddev=1e-2), name='w_k')
        #     key = tf.matmul(self.fc7, w_k)
        #     w_v = tf.Variable(tf.random_normal([4096, 2048],
        #                                         dtype=tf.float32,
        #                                         stddev=1e-2), name='w_v')
        #     value = tf.matmul(self.fc7, w_v)
        #     w_q = tf.Variable(tf.random_normal([4096, 2048],
        #                                         dtype=tf.float32,
        #                                         stddev=1e-2), name='w_q')
        #     query = tf.matmul(self.fc7, w_q)

        #     # [N, D, 1] dot [N, D, 1].T -> [N, D, D] / sqrt(D) 
        #     att = tf.nn.softmax(tf.matmul(tf.expand_dims(key, 2), tf.expand_dims(value, 2), transpose_b=True))
        #     # dot [N, 1, D].T  -> [N, D, 1] -> [N, D]
        #     self.att = tf.reshape(tf.matmul(att, tf.expand_dims(query, 1), transpose_b=True), [-1, 2048])
        #     self.train_layers += [w_k, w_v, w_q]
        
        with tf.name_scope('fc8') as scope:
            fc8w = tf.Variable(tf.random_normal([4096, 300],
                                                dtype=tf.float32,
                                                stddev=1e-2), name='weights')
            fc8b = tf.Variable(tf.constant(0.0, shape=[300],
                                           dtype=tf.float32), name='biases')
            self.fc8l = tf.nn.bias_add(tf.matmul(self.fc7, fc8w), fc8b)
            self.fc8 = tf.nn.tanh(self.fc8l)
            self.deep_param_img['fc8'] = [fc8w, fc8b]
            self.train_last_layer += [fc8w, fc8b]
        # with tf.name_scope('fc8') as scope:
        #     fc8w = tf.Variable(tf.random_normal([4096, 300],
        #                                         dtype=tf.float32,
        #                                         stddev=1e-2), name='weights')
        #     fc8b = tf.Variable(tf.constant(0.0, shape=[4096],
        #                                    dtype=tf.float32), name='biases')
        #     self.fc8l = tf.nn.bias_add(tf.matmul(self.att, fc8w), fc8b)
        #     self.fc8 = tf.nn.dropout(tf.nn.relu(
        #         self.fc8l), 0.5) if self._train else tf.nn.relu(self.fc8l)
        #     self.train_last_layer += [fc8w, fc8b]

        # Classify
        # Output label_num
        with tf.name_scope('cls') as scope:
            clsw = tf.Variable(tf.random_normal([4096, self.n_class],
                                                dtype=tf.float32,
                                                stddev=1e-2), name='weights')
            clsb = tf.Variable(tf.constant(0.0, shape=[self.n_class],
                                           dtype=tf.float32), name='biases')
            self.cls = tf.nn.bias_add(tf.matmul(self.fc7, clsw), clsb)
            self.clsmax = tf.nn.softmax(self.cls)
            self.deep_param_img['cls'] = [clsw, clsb]
            self.classifyLastLayer += [clsw, clsb]

        PrintWithTime(("build model finished: %ds" %
                       (time.time() - start_time)))

    def buildEncoder(self, inp):
        PrintWithTime("Deep Stacked Quantizer")
        PrintWithTime(BarFormat("Integrating NetPQ"))
        self.alexnet(inp)

        self.X = self.fc8

        residual = self.X
        self.HardCode = [None] * self._stackLevel

        D = residual.get_shape().as_list()[-1]
        N = residual.get_shape().as_list()[0]
        # [nlevel, centers, D]
        self.Codebook = tf.Variable(tf.random_normal([self._stackLevel, self._subCenters, D], dtype=tf.float32, stddev=1e-2), name="Codebook")
        # self.Transform = tf.Variable(tf.random_normal([self._stackLevel - 1, D, D], dtype=tf.float32, stddev=1e-2), name="Transform")

        self.QSoft = tf.zeros([N, D])
        self.QHard = tf.zeros([N, D])

        self.SoftDistortion = tf.Variable(0.0, name="soft_distortion")
        self.HardDistortion = tf.Variable(0.0, name="hard_distortion")

        for level in range(self._stackLevel):
            # [K, D]
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

            # [N, K] dot [K, D]
            soft = tf.matmul(tf.nn.softmax(distance), codes)
            code = tf.argmax(distance, axis=1)
            self.HardCode[level] = code
            hard = tf.gather(codes, code)

            residual -= hard

            # if level < self._stackLevel - 1:
            #     trans = tf.gather(self.Transform, level)
            #     residual = residual @ trans

            self.QSoft += soft
            self.QHard += hard

            self.SoftDistortion += tf.reduce_mean(
                tf.square(self.X - self.QSoft))
            self.HardDistortion += tf.reduce_mean(
                tf.square(self.X - self.QHard))

        # self.train_last_layer += [self.Codebook]
        # self.classifyLastLayer += [self.Codebook]

        PrintWithTime("NetPQ output: ")
        print("Qsoft:", self.QSoft.get_shape())
        print("Qhard:", self.QHard.get_shape())

    def ApplyLoss(self, labelInt):
        label = tf.cast(labelInt, tf.float32)
        if self.n_class > 10:
            if self._multiLabel:
                print(NUS_WORD_DICT)
                word_dict = tf.constant(np.loadtxt(NUS_WORD_DICT), dtype=tf.float32)
            else:
                print(IMG_WORD_DICT)
                word_dict = tf.constant(np.loadtxt(IMG_WORD_DICT), dtype=tf.float32)
        else:
            print(CIFAR_WORD_DICT)
            word_dict = tf.constant(np.loadtxt(CIFAR_WORD_DICT), dtype=tf.float32)


        if self.loss_type == 'cos_margin_multi_label':
           # apply L = sum(sum(max(0, delta - cos1 + cos2)))
           # equation (1) in paper
           # hard margin just set delta = constant
           margin_param = tf.constant(self._margin, dtype=tf.float32)

           # N: batch_num, L: label_dim, D: 300
           # img_label: N * L
           # word_dic: L * D
           # v_label: N * L * D
           # the correct label embedding {Vi}

           # label is k-hot (multi-label)
           # the v_label is
           #   [[[ 0,  0,  0],
           #     [ 0,  0,  0],
           #     [-1, -2, -3],
           #     [ 0,  0,  0]],

           #    [[ 2,  3,  4],
           #     [ 0,  0,  0],
           #     [-1, -2, -3],
           #     [ 0,  0,  0]],

           #    [[ 2,  3,  4],
           #     [ 7,  8,  9],
           #     [ 0,  0,  0],
           #     [ 0,  0,  0]]]
           # determine that for x1, the label is [0,0,1,0] and pick the 3rd word vec
           # x2 is [1,0,1,0] and pick the 1st and 3rd word vecs. etc.
           v_label = tf.multiply(tf.expand_dims(
               label, 2), tf.expand_dims(word_dict, 0))

           # img_last: N * D
           # ip_1: N * L
           # dot product: < {Vi}.T, Z >, here has broadcasting
           # [N, 1, D] * [N, L, D] = [N, L, D], sum -> [N, L]
           ip_1 = tf.reduce_sum(tf.multiply(
               tf.expand_dims(self.fc8, 1), v_label), 2)

           # mod_1: N * L
           # || Vi || * || Z ||
           v_label_mod = tf.multiply(tf.expand_dims(
               tf.ones([self.batch_size, self.n_class]), 2), tf.expand_dims(word_dict, 0))
           mod_1 = tf.sqrt(tf.multiply(tf.expand_dims(tf.reduce_sum(
               tf.square(self.fc8), 1), 1), tf.reduce_sum(tf.square(v_label_mod), 2)))

           # mod_1 = tf.where(tf.less(mod_1_1, tf.constant(0.0000001)),
           # tf.ones([self.batch_size, self.n_class]), mod_1_1)
           # cos_1: N * L
           cos_1 = tf.div(ip_1, mod_1)

           # all label embedding {V}
           ip_2 = tf.matmul(self.fc8, word_dict, transpose_b=True)

           # multiply ids to inner product
           # ip_2 = tf.multiply(ip_2_1, ids_dict)

           def reduce_shaper(t):
               return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])

           # same calculation as mod_1, 很迷的操作
           mod_2_2 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(self.fc8)), reduce_shaper(
               tf.square(word_dict)), transpose_b=True))
           # pick where label is 1, set them to 0, 相当于去掉对的词
           mod_2 = tf.where(tf.less(mod_2_2, tf.constant(0.0000001)), tf.ones(
               [self.batch_size, self.n_class]), mod_2_2)
           # cos_2: N * L
           cos_2 = tf.div(ip_2, mod_2)

           # cos - cos: N * L * L
           # delta - cos1 + cos2
           cos_cos_1 = tf.subtract(margin_param, tf.subtract(
               tf.expand_dims(cos_1, 2), tf.expand_dims(cos_2, 1)))
           # we need to let the wrong place be 0
           # only use i∈Y, as in the first sum
           cos_cos = tf.multiply(cos_cos_1, tf.expand_dims(label, 2))
           # sum up
           cos_loss = tf.reduce_sum(tf.maximum(
               tf.constant(0, dtype=tf.float32), cos_cos))
           # average them here is the total num of sample
           self.cos_loss = tf.div(cos_loss, tf.multiply(tf.constant(
               self.n_class, dtype=tf.float32), tf.reduce_sum(label)))

        elif self.loss_type == 'cos_softmargin_multi_label':
           # N: batchsize, L: label_dim, D: 300
           # img_label: N * L
           # word_dic: L * D
           # v_label: N * L * D
           v_label = tf.multiply(tf.expand_dims(label, 2), tf.expand_dims(word_dict, 0))
           # img_last: N * D
           # ip_1: N * L
           ip_1 = tf.reduce_sum(tf.multiply(tf.expand_dims(self.fc8, 1), v_label), 2)
           # mod_1: N * L
           v_label_mod = tf.multiply(tf.expand_dims(tf.ones([self.batch_size, self.n_class]), 2), tf.expand_dims(word_dict, 0))
           mod_1 = tf.sqrt(tf.multiply(tf.expand_dims(tf.reduce_sum(tf.square(self.fc8), 1), 1), tf.reduce_sum(tf.square(v_label_mod), 2)))
           # mod_1 = tf.where(tf.less(mod_1_1, tf.constant(0.0000001)),
           # tf.ones([self.batch_size, self.n_class]), mod_1_1)
           # cos_1: N * L
           cos_1 = tf.div(ip_1, mod_1)

           ip_2 = tf.matmul(self.fc8, word_dict, transpose_b=True)

           # multiply ids to inner product
           # ip_2 = tf.multiply(ip_2_1, ids_dict)

           def reduce_shaper(t):
               return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])

           mod_2_2 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(self.fc8)), reduce_shaper(
               tf.square(word_dict)), transpose_b=True))
           mod_2 = tf.where(tf.less(mod_2_2, tf.constant(0.0000001)), tf.ones(
               [self.batch_size, self.n_class]), mod_2_2)
           # cos_2: N * L
           cos_2 = tf.div(ip_2, mod_2)

           # word_dic: L * D
           # ip_3: L * L
           # compute soft margin
           ip_3 = tf.matmul(word_dict, word_dict, transpose_b=True)
           # use word_dic to avoid 0 in /
           mod_3 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(word_dict)), reduce_shaper(
               tf.square(word_dict)), transpose_b=True))

           # soft_margin is explained as paper
           margin_param = 1 - (ip_3 / mod_3)

           # cos - cos: N * L * L
           cos_cos_1 = tf.subtract(tf.expand_dims(margin_param, 0), tf.subtract(
               tf.expand_dims(cos_1, 2), tf.expand_dims(cos_2, 1)))
           # we need to let the wrong place be 0
           cos_cos = tf.multiply(cos_cos_1, tf.expand_dims(label, 2))

           cos_loss = tf.reduce_sum(tf.maximum(
               tf.constant(0, dtype=tf.float32), cos_cos))
           self.cos_loss = tf.div(cos_loss, tf.multiply(tf.constant(
               self.n_class, dtype=tf.float32), tf.reduce_sum(label)))

        self.classify = self._lambda * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls, labels=label) if self._multiLabel == False else tf.nn.sigmoid_cross_entropy_with_logits(logits=self.cls, labels=label))
        print("Multi Label:", self._multiLabel)

        self.loss = self.cos_loss + self.classify
        """ Quantization Loss """
        # JCL
        self.JointCenter = tf.reduce_mean(tf.square(self.QSoft-self.QHard))
        # Distortion summarized at Inference
        PrintWithTime(BarFormat("Loss built"))
