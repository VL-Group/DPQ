#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function, division

import errno
import os
import random
from datetime import datetime

import numpy as np


def PrintWithTime(string=None):
    if string is None:
        print("{0}".format(datetime.now().strftime("%m-%d %H:%M:%S")))
    else:
        print("{0}  {1}".format(datetime.now().strftime("%m-%d %H:%M:%S"), string))


def ProgressBar(progress):
    progress *= 100
    progress = round(progress)
    a = '=' * int(progress) + '-' * (100 - int(progress))
    a = a[:47] + (' 100 ' if progress >= 100 else " %2d%% " % progress) + a[52:] + '\n'
    print(a)


def BarFormat(string):
    l = len(string)
    a = (100 - 2 - l) // 2
    b = 2 * a + 2 + l == 99
    return '=' * a + ' {0} '.format(string) + '=' * (a + 1) if b else '=' * a + ' {0} '.format(string) + '=' * a


def cosine_distance(u, v):
    # [N1, D] dot [D, N2] -> [N1, N2]
    ux, vy = np.meshgrid(np.linalg.norm(u, 2, 1), np.linalg.norm(v, 2, 1))
    return 1 - (u.dot(v.T) / (ux * vy).T)


class mAP:
    def __init__(self, C, R, database):
        """[summary]

        [description]

        Parameters
        ----------
        C : ndarray
            Codebook, [M, K, d],
            the codeword is M * [0 ~ K - 1],
            for getting quantized result:
                first get the subspace results: C[range(M), codeword] -> [M, d]
                then concate -> [D, ]
        R : int
            retrieve top R results, set it same as query count
        database: Object
            label: the class label
            codes: the codeword, [N, M]
            output: the feature vector, [N, D]
        """
        # Codebook, M * center_num
        self.C = C
        # retrieve top R results
        self.R = R
        self._database = database

    def calculatemAP(self, query_labels, distance):
        # sort by distance for each query
        ids = np.argsort(distance, 1)
        database_labels = self._database.label
        APx = []
        for i in range(distance.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1
            idx = ids[i, :]
            imatch = np.sum(database_labels[idx[0: self.R], :] == label, 1) > 0
            rel = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
            if rel != 0:
                APx.append(np.sum(Px * imatch) / rel)
            else:
                APx.append(0.0)
        return APx

    @staticmethod
    # [N, M], [M, K, d]
    def Quantize_PQ(codes, codebook):
        M, k, d = codebook.shape
        # [N, K, d]
        q = codebook[range(M), codes[:]]
        # [N, D]
        return q.reshape([codes.shape[0], -1])

    @staticmethod
    # [N, M], [M, K, d]
    def Quantize_AQ(codes, codebook, level):
        M, k, D = codebook.shape
        assert M >= level, "Quantize level out of range"
        # [N, D]
        q = np.zeros([codes.shape[0], D])
        for i in range(level):
            print('Codebook %d' % i)
            q += codebook[i, codes[:, i]]
        # [N, D]
        return q

    def SQD_mAP(self, query):
        # all the distance between query and data
        # using quantized - quantized

        # [N, D]
        db = self.Quantize_AQ(self._database.codes, self.C)
        qy = self.Quantize_AQ(query.codes, self.C)
        print('distance')

        APx = np.zeros([qy.shape[0]])

        for j in range(qy.shape[0] // 50 + 1):
            q = qy[j * 50:(j + 1) * 50]
            d = -np.dot(q, db.T)
            ids = np.argsort(d, 1)

            for i in range(d.shape[0]):
                label = query.label[j * 50 + i, :]
                label[label == 0] = -1
                imatch = np.sum(self._database.label[ids[i, :self.R], :] == label, 1) > 0
                rel = np.sum(imatch)
                Lx = np.cumsum(imatch)
                Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
                if rel != 0:
                    APx[j * 50 + i] = np.sum(Px * imatch) / rel
                else:
                    APx[j * 50 + i] = 0

            print("%d / %d" % (j, qy.shape[0] // 50 + 1))

        result = np.mean(APx)
        PrintWithTime("SQD mAPs: " + str(result))
        return np.asscalar(result)

    def AQD_mAP(self, query):
        # all the distance between query and data
        # using actual - quantized
        print(self.R)
        print(self._database.codes.shape[0])

        for k in [4, 3, 2, 1]:
            # [N, D]
            db = self.Quantize_AQ(self._database.codes, self.C, k)

            print('distance')

            APx = np.zeros([query.output.shape[0]])

            ids = np.argsort(-query.output.dot(db.T), 1)
            query.label[query.label == 0] = -1
            imatchs = np.sum(self._database.label[ids[:, :self.R]] == np.expand_dims(query.label, 1), 2) > 0
            rel = np.sum(imatchs, 1)
            Px = np.cumsum(imatchs, 1).astype(float) / np.arange(1, self.R + 1, 1)
            rel[rel == 0] = -1
            APx = np.sum(Px * imatchs, 1) / rel
            APx[APx < 0] = 0
            # for i in range(d.shape[0]):
            #     label = query.label[j*50 + i, :]
            #     label[label == 0] = -1
            #     idx = ids[i, :]
            #     imatch = np.sum(self._database.label[idx[0: self.R], :] == label, 1) > 0
            #     rel = np.sum(imatch)
            #     Lx = np.cumsum(imatch)
            #     Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
            #     if rel != 0:
            #         APx[j*50+i] = np.sum(Px * imatch) / rel
            #     else:
            #         APx[j*50+i] = 0
            # print("%d / %d" % (k, query.output.shape[0] // 50 + 1))

            result = np.mean(APx)
            print("Quantize level %d, AQD mAP@%d: %f" % (k, self.R, result))
            del result
            del db
            del APx

    def Feature_mAP(self, query):
        # all the distance between query and data
        # using actual - actual

        print('distance')

        APx = np.zeros([query.output.shape[0]])

        for j in range(query.output.shape[0] // 50 + 1):
            q = query.output[j * 50:(j + 1) * 50]
            d = -np.dot(q, self._database.output.T)
            ids = np.argsort(d, 1)
            for i in range(d.shape[0]):
                label = query.label[j * 50 + i, :]
                label[label == 0] = -1
                idx = ids[i, :]
                imatch = np.sum(self._database.label[idx[0: self.R], :] == label, 1) > 0
                rel = np.sum(imatch)
                Lx = np.cumsum(imatch)
                Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
                if rel != 0:
                    APx[j * 50 + i] = np.sum(Px * imatch) / rel
                else:
                    APx[j * 50 + i] = 0

            print("%d / %d" % (j, query.output.shape[0] // 50 + 1))

        result = np.mean(APx)

        # print('l2 distance')

        # APx = np.zeros([query.output.shape[0]])

        # for j in range(query.output.shape[0] // 50 + 1):
        #     q = query.output[j*50:(j+1)*50]
        #     d = L2Distance(q, self._database.output)
        #     ids = np.argsort(d, 1)
        #     for i in range(d.shape[0]):
        #         label = query.label[j*50 + i, :]
        #         label[label == 0] = -1
        #         idx = ids[i, :]
        #         imatch = np.sum(self._database.label[idx[0: self.R], :] == label, 1) > 0
        #         rel = np.sum(imatch)
        #         Lx = np.cumsum(imatch)
        #         Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
        #         if rel != 0:
        #             APx[j*50+i] = np.sum(Px * imatch) / rel
        #         else:
        #             APx[j*50+i] = 0

        #     print("%d / %d" % (j, query.output.shape[0] // 50 + 1))

        # result_l2 = np.mean(APx)

        print("Feature mAP@%d: %f" % (self.R, result))
        # PrintWithTime("Feature mAP (l2 distance): " + str(result_l2))
        return np.asscalar(result)


class Object(object):
    pass


def L2Distance(a, b):
    """square distance of mat A and B

    compute Euclidean distance of A and B into output mat

    Parameters
    ----------
    a : ndarray
        A [k, d]
    b : ndarray
        B [N, d]
    
    Returns
    -------
    ndarray, [k, N]
    """
    # [N, k]
    d = -2 * b.dot(a.T) + np.sum(np.square(b), axis=1, keepdims=True)
    # [k, N]
    d = np.abs(d.T + np.sum(np.square(a), axis=1, keepdims=True))
    # [k, N]
    return d


def CreateFile(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def RandInt(num):
    return int(random.random() * num)


def CountVariables(var_list):
    return np.sum([np.prod(v.get_shape().as_list()) for v in var_list])
