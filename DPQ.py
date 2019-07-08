#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function, division

import os
import time

import numpy as np
import tensorflow as tf

from Dataset import Dataset
from Encoder_Alex import Encoder_Alex
from Encoder_VGG import Encoder_VGG
from Utils import PrintWithTime, ProgressBar, BarFormat, CountVariables, Object, mAP

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CNN_STR = ["Alex", "VGG"]

CNN_TYPE = 0

SESSION_SAVE_PATH = "./DSQ_{}.ckpt".format(CNN_STR[CNN_TYPE])

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


class DPQ(object):
    def __init__(self, FLAG):
        # used for prediction (classification)
        self._classNum = FLAG.ClassNum
        # center matrix C: [M * K * D]
        # D = U, U is the embedding layer output dimension
        self._k = FLAG.K
        # from code length get sub space count
        assert self._k != 0 and (self._k & (self._k - 1)) == 0
        perLength = int(np.asscalar(np.log2(self._k)))
        self._stackLevel = FLAG.BitLength // perLength

        PrintWithTime("Init with config:")
        print("                # Stack Levels :", self._stackLevel)
        print("                # Class Num  :", self._classNum)
        print("                # Centers K  :", self._k)

        # other settings for learning
        self._initLR = FLAG.LearningRate
        self._epoch = FLAG.Epoch
        self._batchSize = FLAG.BatchSize
        self._saveModel = FLAG.SaveModel
        self._recallatR = FLAG.R
        self._multiLabel = FLAG.Dataset == "NUS"

        self._lambda = FLAG.Lambda
        self._tau = FLAG.Tau
        self._mu = FLAG.Mu
        self._nu = FLAG.Nu

        # other settings for printing
        self._printEvery = FLAG.PrintEvery

        assert (FLAG.Mode == 'train' or FLAG.Mode == 'eval')

        self._train = FLAG.Mode == 'train'

        if self._train:
            # dataset
            self.Dataset = Dataset(FLAG.Dataset, FLAG.Mode, FLAG.BatchSize,
                                   IMAGE_WIDTH, IMAGE_HEIGHT)
        self.DatasetName = FLAG.Dataset
        # tensorflow configs
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self._config = config

        self.NetPQ = Encoder_Alex(self._batchSize, self._classNum, self._lambda, self._stackLevel, self._k,
                                  self._multiLabel, self._train) if CNN_TYPE == 0 else Encoder_VGG(self._batchSize,
                                                                                                   self._classNum,
                                                                                                   self._stackLevel,
                                                                                                   self._k, self._train)

        self._name = "lr_{0}_batch_{1}_M_{2}_K_{3}".format(self._initLR, self._batchSize, self._stackLevel, self._k)

    def Inference(self):
        self.NetPQ.Inference(self.Input, self.LabelHot)

    def ApplyLoss(self):
        lr = tf.train.exponential_decay(self._initLR, global_step=self.GlobalStep, decay_steps=10000, decay_rate=0.9)
        codebooklr = tf.train.exponential_decay(1e-4, global_step=self.GlobalStep, decay_steps=10000, decay_rate=0.9)

        print("Total var num:", CountVariables(tf.trainable_variables()))

        # Note that these are updated respectively
        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        g_1 = opt.compute_gradients(self.NetPQ.cos_loss, self.NetPQ.train_layers + self.NetPQ.train_last_layer)
        main_fcgrad, _ = g_1[-2]
        main_fbgrad, _ = g_1[-1]

        g_2 = opt.compute_gradients(self.NetPQ.classify, self.NetPQ.train_layers + self.NetPQ.classifyLastLayer)
        cls_fcgrad, _ = g_2[-2]
        cls_fbgrad, _ = g_2[-1]
        if CNN_TYPE == 0:
            self.TrainEncoder_FINE_TUNE = opt.apply_gradients(
                [((g_1[0][0] + g_2[0][0]) / 2, self.NetPQ.train_layers[0]),
                 (g_1[1][0] + g_2[1][0],
                  self.NetPQ.train_layers[1]),
                 ((g_1[2][0] + g_2[2][0]) / 2,
                  self.NetPQ.train_layers[2]),
                 (g_1[3][0] + g_2[3][0],
                  self.NetPQ.train_layers[3]),
                 ((g_1[4][0] + g_2[4][0]) / 2,
                  self.NetPQ.train_layers[4]),
                 (g_1[5][0] + g_2[5][0],
                  self.NetPQ.train_layers[5]),
                 ((g_1[6][0] + g_2[6][0]) / 2,
                  self.NetPQ.train_layers[6]),
                 (g_1[7][0] + g_2[7][0],
                  self.NetPQ.train_layers[7]),
                 ((g_1[8][0] + g_2[8][0]) / 2,
                  self.NetPQ.train_layers[8]),
                 (g_1[9][0] + g_2[9][0],
                  self.NetPQ.train_layers[9]),
                 ((g_1[10][0] + g_2[10][0]) / 2,
                  self.NetPQ.train_layers[10]),
                 (g_1[11][0] + g_2[11][0],
                  self.NetPQ.train_layers[11]),
                 ((g_1[12][0] + g_2[12][0]) / 2,
                  self.NetPQ.train_layers[12]),
                 (g_1[13][0] + g_2[13][0],
                  self.NetPQ.train_layers[13]),
                 (main_fcgrad * 10,
                  self.NetPQ.train_last_layer[0]),
                 (main_fbgrad * 20,
                  self.NetPQ.train_last_layer[1]),
                 (cls_fcgrad * 10,
                  self.NetPQ.classifyLastLayer[0]),
                 (cls_fbgrad * 20, self.NetPQ.classifyLastLayer[1]), ],
                global_step=self.GlobalStep)
        else:
            self.TrainEncoder_FINE_TUNE = opt.apply_gradients(
                [((g_1[0][0] + g_2[0][0]) / 2, self.NetPQ.train_layers[0]),
                    (g_1[1][0] + g_2[1][0], self.NetPQ.train_layers[1]),
                    ((g_1[2][0] + g_2[2][0]) / 2, self.NetPQ.train_layers[2]),
                    (g_1[3][0] + g_2[3][0], self.NetPQ.train_layers[3]),
                    ((g_1[4][0] + g_2[4][0]) / 2, self.NetPQ.train_layers[4]),
                    (g_1[5][0] + g_2[5][0], self.NetPQ.train_layers[5]),
                    ((g_1[6][0] + g_2[6][0]) / 2, self.NetPQ.train_layers[6]),
                    (g_1[7][0] + g_2[7][0], self.NetPQ.train_layers[7]),
                    ((g_1[8][0] + g_2[8][0]) / 2, self.NetPQ.train_layers[8]),
                    (g_1[9][0] + g_2[9][0], self.NetPQ.train_layers[9]),
                    ((g_1[10][0] + g_2[10][0]) / 2,
                    self.NetPQ.train_layers[10]),
                    (g_1[11][0] + g_2[11][0], self.NetPQ.train_layers[11]),
                    ((g_1[12][0] + g_2[12][0]) / 2,
                    self.NetPQ.train_layers[12]),
                    (g_1[13][0] + g_2[13][0], self.NetPQ.train_layers[13]),
                    ((g_1[14][0] + g_2[14][0]) / 2,
                    self.NetPQ.train_layers[14]),
                    (g_1[15][0] + g_2[15][0], self.NetPQ.train_layers[15]),
                    ((g_1[16][0] + g_2[16][0]) / 2,
                    self.NetPQ.train_layers[16]),
                    (g_1[17][0] + g_2[17][0], self.NetPQ.train_layers[17]),
                    ((g_1[18][0] + g_2[18][0]) / 2,
                    self.NetPQ.train_layers[18]),
                    (g_1[19][0] + g_2[19][0], self.NetPQ.train_layers[19]),
                    ((g_1[20][0] + g_2[20][0]) / 2,
                    self.NetPQ.train_layers[20]),
                    (g_1[21][0] + g_2[21][0], self.NetPQ.train_layers[21]),
                    ((g_1[22][0] + g_2[22][0]) / 2,
                    self.NetPQ.train_layers[22]),
                    (g_1[23][0] + g_2[23][0], self.NetPQ.train_layers[23]),
                    ((g_1[24][0] + g_2[24][0]) / 2,
                    self.NetPQ.train_layers[24]),
                    (g_1[25][0] + g_2[25][0], self.NetPQ.train_layers[25]),
                    ((g_1[26][0] + g_2[26][0]) / 2,
                    self.NetPQ.train_layers[26]),
                    (g_1[27][0] + g_2[27][0], self.NetPQ.train_layers[27]),
                    ((g_1[28][0] + g_2[28][0]) / 2,
                    self.NetPQ.train_layers[28]),
                    (g_1[29][0] + g_2[29][0], self.NetPQ.train_layers[29]),
                    (main_fcgrad * 10, self.NetPQ.train_last_layer[0]),
                    (main_fbgrad * 20, self.NetPQ.train_last_layer[1]),
                    (cls_fcgrad * 10, self.NetPQ.classifyLastLayer[0]),
                    (cls_fbgrad * 20, self.NetPQ.classifyLastLayer[1])],
                    global_step=self.GlobalStep)

        self.TrainCodebook = tf.train.AdamOptimizer(learning_rate=codebooklr).minimize(self._tau * (
                    self.NetPQ.SoftDistortion + self._mu * self.NetPQ.HardDistortion + self._nu * self.NetPQ.JointCenter),
                                                                                       global_step=self.GlobalStep,
                                                                                       var_list=[self.NetPQ.Codebook])

    def InitVariables(self):
        self.Input = tf.placeholder(tf.float32, shape=[self._batchSize, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="Input")
        self.LabelHot = tf.placeholder(tf.int32, shape=[self._batchSize, self._classNum], name="Label")

        self.GlobalStep = tf.Variable(0, trainable=False)

        self.Inference()
        self.ApplyLoss()

        PrintWithTime(BarFormat("Variables Inited"))

    def AddSummary(self, graph):
        tf.summary.scalar('Semantic Loss', self.NetPQ.cos_loss)
        tf.summary.scalar('Classification Loss', self.NetPQ.classify)
        tf.summary.scalar('Soft Distortion', self.NetPQ.SoftDistortion)
        tf.summary.scalar('Hard Distortion', self.NetPQ.HardDistortion)
        tf.summary.scalar('JCL', self.NetPQ.JointCenter)
        tf.summary.histogram('Codebook', self.NetPQ.Codebook)

        # Merge all the summaries and write them out to /tmp/mnist_logs (by
        # default)
        self._summary = tf.summary.merge_all()
        self._writer = tf.summary.FileWriter('/tmp/DPQ', graph=graph)

    def Train(self):
        PrintWithTime(BarFormat("Training Start"))

        start = time.time()

        with tf.Session(config=self._config) as sess:
            sess.run(tf.global_variables_initializer())
            if self._saveModel:
                # Create a saver
                self._saver = tf.train.Saver()
            self.AddSummary(sess.graph)

            """ Pre-train stage """
            PrintWithTime(BarFormat("Pre-train Stage"))
            for i in range(self._epoch // 2):
                j = 0
                if self._saveModel:
                    self._saver.save(sess, SESSION_SAVE_PATH)
                PrintWithTime("Saved @ epoch {0}".format(i))
                while not self.Dataset.EpochComplete:
                    j += 1
                    image, label = self.Dataset.NextBatch()
                    assert image.shape[0] == self._batchSize
                    _ = sess.run(self.TrainEncoder_FINE_TUNE,
                                 {self.Input: image, self.LabelHot: label})
                    if j % self._printEvery == 0:
                        # Can't simply run with (self.NetPQ.JointCenter +
                        # self.NetPQ.Distortion + self.NetPQ.QHard +
                        # self.NetPQ.QSoft)
                        # This will cause graph re-creation and variables
                        # re-allocation
                        PrintWithTime("Epoch {0}, Step {1}: total loss = {2}".format(i, j, np.mean(
                            sess.run(self.NetPQ.loss, {self.Input: image, self.LabelHot: label}))))
                        ProgressBar((i + self.Dataset.Progress) / self._epoch)
                        # self._writer.add_summary(sess.run(self._summary, {self.Input: image, self.LabelHot: label}), global_step=self.GlobalStep)

            """ Codebook learning stage """
            PrintWithTime(BarFormat("Codebook Learning Stage"))
            for i in range(self._epoch // 2 + 1, self._epoch):
                if self._saveModel:
                    self._saver.save(sess, SESSION_SAVE_PATH)
                PrintWithTime("Saved @ epoch {0}".format(i))
                while not self.Dataset.EpochComplete:
                    j += 1
                    image, label = self.Dataset.NextBatch()
                    assert image.shape[0] == self._batchSize
                    _ = sess.run(self.TrainEncoder_FINE_TUNE,
                                 {self.Input: image, self.LabelHot: label})
                    _ = sess.run(self.TrainCodebook,
                                 {self.Input: image, self.LabelHot: label})

                    if j % self._printEvery == 0:
                        jointLoss, hardDistLoss, softDistLoss, netPQLoss = sess.run(
                            [self.NetPQ.JointCenter, self.NetPQ.HardDistortion, self.NetPQ.SoftDistortion,
                             self.NetPQ.loss],
                            {self.Input: image, self.LabelHot: label})
                        net = [np.mean(jointLoss), np.mean(softDistLoss), np.mean(hardDistLoss), np.mean(netPQLoss)]
                        PrintWithTime("Epoch {0}, Step {1}: NetPQ Loss: {2}".format(i, j, net))
                        ProgressBar((i + self.Dataset.Progress) / self._epoch)
                        # self._writer.add_summary(sess.run(self._summary, {self.Input: image, self.LabelHot: label}), global_step=self.GlobalStep)
            end = time.time()
            print('%d seconds for %d epochs, %d batches and %d samples' % (end - start, self._epoch, j, j * self._batchSize))
            PrintWithTime(BarFormat("Train Finished"))

    def Evaluate(self, queryX, queryY, dataset):
        print(self._recallatR if self._recallatR > 0 else 'all')
        if os.path.exists(SESSION_SAVE_PATH + '.meta'):
            with tf.Session(config=self._config) as sess:
                self.InitVariables()
                self._saver = tf.train.Saver()
                self._saver.restore(sess, SESSION_SAVE_PATH)
                PrintWithTime("Restored model from " + SESSION_SAVE_PATH)

                query = Object()
                database = Object()
                query.label = queryY

                Nq = queryX.shape[0]

                dim = self.NetPQ.X.get_shape().as_list()[1]

                query.output = np.zeros([Nq, dim], np.float16)
                for i in range((Nq // self._batchSize) + 1):
                    inp = queryX[i * self._batchSize:(i + 1) * self._batchSize]
                    num = inp.shape[0]
                    if inp.shape[0] != self._batchSize:
                        placeholder = np.zeros(
                            [self._batchSize - inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]])
                        inp = np.concatenate((inp, placeholder))

                    out = sess.run(self.NetPQ.X, {self.Input: inp})
                    query.output[i * self._batchSize:(i * self._batchSize) + num] = out[:num]

                Nb = dataset.DataNum
                database_feature = np.zeros([Nb, dim], dtype=np.float16)
                database.label = np.zeros([Nb, self._classNum], dtype=np.int16)

                database.codes = np.zeros([Nb, self._stackLevel], np.int32)

                start = time.time()
                print('Encoding database')
                total_db = (Nb // self._batchSize) + 1
                for i in range(total_db):
                    idx = np.arange(start=i * self._batchSize,
                                    stop=np.minimum(Nb, (i + 1) * self._batchSize), step=1)
                    inp, label = dataset.Get(idx)
                    print(inp.shape, label.shape)
                    num = inp.shape[0]
                    database.label[i * self._batchSize:(i * self._batchSize + num)] = label
                    if inp.shape[0] != self._batchSize:
                        placeholder = np.zeros(
                            [self._batchSize - inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]])
                        inp = np.concatenate((inp, placeholder))

                    hardCode = sess.run(self.NetPQ.HardCode, {self.Input: inp})
                    database.codes[i * self._batchSize:(i * self._batchSize) + num] = np.array(hardCode, np.int32).T[
                                                                                      :num]
                    database_feature[i * self._batchSize:(i * self._batchSize) + num] = out[:num]
                    ProgressBar((i + 1) / total_db)

                end = time.time()
                print('Encoding Complete')
                print('Time:', end - start)
                print('Average time for single sample:')
                print((end - start) / Nb)
                database.output = database_feature

                del dataset

                codebook = sess.run(self.NetPQ.Codebook)

                res = mAP(codebook, self._recallatR if self._recallatR > 0 else database.codes.shape[0], database)

                return res.AQD_mAP(query)

    def CheckTime(self, queryX):
        if os.path.exists(SESSION_SAVE_PATH + '.meta'):
            with tf.Session(config=self._config) as sess:
                self.InitVariables()
                self._saver = tf.train.Saver()
                self._saver.restore(sess, SESSION_SAVE_PATH)
                PrintWithTime("Restored model from " + SESSION_SAVE_PATH)

                inp = queryX[:self._batchSize]

                start = time.time()
                for _ in range(1000):
                    _ = sess.run(self.NetPQ.HardCode, {self.Input: inp})
                end = time.time()
                print('total time', end - start)
                print('avg time', (end - start) / (1000 * self._batchSize))

    def EvalClassification(self, queryX, queryY):
        if os.path.exists(SESSION_SAVE_PATH + '.meta'):
            with tf.Session(config=self._config) as sess:
                self.InitVariables()
                self._saver = tf.train.Saver()
                self._saver.restore(sess, SESSION_SAVE_PATH)
                PrintWithTime("Restored model from " + SESSION_SAVE_PATH)
                Nq = queryX.shape[0]
                dim = self._classNum

                if self.DatasetName == 'NUS':
                    result = -1 * np.ones([Nq, dim], np.int)

                    for i in range((Nq // self._batchSize) + 1):
                        inp = queryX[i * self._batchSize:(i + 1) * self._batchSize]
                        num = inp.shape[0]
                        if inp.shape[0] != self._batchSize:
                            placeholder = np.zeros(
                                [self._batchSize - inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]])
                            inp = np.concatenate((inp, placeholder))

                        out = sess.run(self.NetPQ.cls, {self.Input: inp})
                        for j in range(num):
                            result[i * self._batchSize + j, np.argsort(out[j])[::-1][:2]] = 1

                    checked = np.sum(np.equal(result, queryY), axis=1) > 0
                    accuracy = np.mean(checked)
                    print(accuracy)
                    return

                result = np.zeros([Nq], np.int)

                for i in range((Nq // self._batchSize) + 1):
                    inp = queryX[i * self._batchSize:(i + 1) * self._batchSize]
                    num = inp.shape[0]
                    if inp.shape[0] != self._batchSize:
                        placeholder = np.zeros(
                            [self._batchSize - inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]])
                        inp = np.concatenate((inp, placeholder))

                    out = sess.run(self.NetPQ.cls, {self.Input: inp})
                    result[i * self._batchSize:(i * self._batchSize) + num] = np.argmax(out[:num], axis=1)

                accuracy = np.mean(np.equal(result, np.argmax(queryY, axis=1)))
                print(accuracy)

    def GetRetrievalMat(self, queryX, queryY, dataset):
        self.R = self._recallatR if self._recallatR > 0 else dataset.DataNum
        if os.path.exists(SESSION_SAVE_PATH + '.meta'):
            with tf.Session(config=self._config) as sess:
                self.InitVariables()
                self._saver = tf.train.Saver()
                self._saver.restore(sess, SESSION_SAVE_PATH)
                PrintWithTime("Restored model from " + SESSION_SAVE_PATH)

                query = Object()
                database = Object()
                query.label = queryY

                Nq = queryX.shape[0]

                dim = self.NetPQ.X.get_shape().as_list()[1]

                query_feature = np.zeros([Nq, dim], np.float16)
                for i in range((Nq // self._batchSize) + 1):
                    inp = queryX[i * self._batchSize:(i + 1) * self._batchSize]
                    num = inp.shape[0]
                    if inp.shape[0] != self._batchSize:
                        placeholder = np.zeros(
                            [self._batchSize - inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]])
                        inp = np.concatenate((inp, placeholder))

                    out = sess.run(self.NetPQ.X, {self.Input: inp})
                    query_feature[i * self._batchSize:(i * self._batchSize) + num] = out[:num]
                query.output = query_feature

                Nb = dataset.DataNum
                database.label = np.zeros([Nb, self._classNum], dtype=np.int16)

                codes = np.zeros([Nb, self._stackLevel], np.int32)

                total_db = (Nb // self._batchSize) + 1
                for i in range(total_db):
                    idx = np.arange(start=i * self._batchSize,
                                    stop=np.minimum(Nb, (i + 1) * self._batchSize), step=1)
                    inp, label = dataset.Get(idx)
                    print(inp.shape, label.shape)
                    num = inp.shape[0]
                    database.label[i * self._batchSize:(i * self._batchSize + num)] = label
                    if inp.shape[0] != self._batchSize:
                        placeholder = np.zeros(
                            [self._batchSize - inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]])
                        inp = np.concatenate((inp, placeholder))

                    hardCode = sess.run(self.NetPQ.HardCode, {self.Input: inp})
                    codes[i * self._batchSize:(i * self._batchSize) + num] = np.array(hardCode, np.int32).T[:num]
                    ProgressBar((i + 1) / total_db)

                # [N, M]
                database.codes = codes
                codebook = sess.run(self.NetPQ.Codebook)
                # np.save('database_codes_DSQ', codes)
            db = mAP.Quantize_AQ(database.codes, codebook, 4).T

            del dataset
            id_all = np.zeros([query.output.shape[0], self.R], np.int)
            retrieval_mat = np.zeros([query.output.shape[0], self.R], np.bool)
            for j in range(query.output.shape[0] // 50 + 1):
                q = query.output[j * 50:(j + 1) * 50]
                d = -np.dot(q, db)
                ids = np.argsort(d, 1)
                for i in range(d.shape[0]):
                    label = query.label[j * 50 + i, :]
                    label[label == 0] = -1
                    idx = ids[i, :]
                    imatch = np.sum(database.label[idx[0: self.R], :] == label, 1) > 0
                    id_all[j * 50 + i] = idx[:self.R]
                    retrieval_mat[j * 50 + i] = imatch[:self.R]
            np.save('retrievalMat_' + self.DatasetName, retrieval_mat)
            np.save('ids', id_all)
            return retrieval_mat, id_all

    def GetFeature(self, dataset):
        if os.path.exists(SESSION_SAVE_PATH + '.meta'):
            with tf.Session(config=self._config) as sess:
                self.InitVariables()
                self._saver = tf.train.Saver()
                self._saver.restore(sess, SESSION_SAVE_PATH)
                PrintWithTime("Restored model from " + SESSION_SAVE_PATH)
                database = Object()

                dim = self.NetPQ.X.get_shape().as_list()[1]

                Nb = dataset.DataNum
                database_feature = np.zeros([Nb, dim], dtype=np.float16)
                database.label = np.zeros([Nb, self._classNum], dtype=np.int16)

                codes = np.zeros([Nb, self._stackLevel], np.int32)

                total_db = (Nb // self._batchSize) + 1
                for i in range(total_db):
                    idx = np.arange(start=i * self._batchSize,
                                    stop=np.minimum(Nb, (i + 1) * self._batchSize), step=1)
                    inp, label = dataset.Get(idx)
                    print(inp.shape, label.shape)
                    num = inp.shape[0]
                    database.label[i * self._batchSize:(i * self._batchSize + num)] = label
                    if inp.shape[0] != self._batchSize:
                        placeholder = np.zeros(
                            [self._batchSize - inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]])
                        inp = np.concatenate((inp, placeholder))

                    out, hardCode = sess.run([self.NetPQ.X, self.NetPQ.HardCode], {self.Input: inp})
                    hardCode = sess.run(self.NetPQ.HardCode, {self.Input: inp})
                    codes[i * self._batchSize:(i * self._batchSize) + num] = np.array(hardCode, np.int32).T[:num]
                    database_feature[i * self._batchSize:(i * self._batchSize) + num] = out[:num]
                    ProgressBar((i + 1) / total_db)
                database.output = database_feature

                # [N, M]
                database.codes = codes
                codebook = sess.run(self.NetPQ.Codebook)
            return database, codebook

    def Save(self):
        with tf.Session(config=self._config) as sess:
            # Save the session
            save_path = self._saver.save(sess, SESSION_SAVE_PATH)
            PrintWithTime(BarFormat("Model saved"))
            PrintWithTime("Path: " + save_path)
