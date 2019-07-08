#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function, division

import pickle as p
import sys
from os import path

import cv2
import numpy as np

py3 = sys.version_info >= (3, 4)

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
LABEL_NAME = ["airplane", "automobile", "bird", "cat",
              "deer", "dog", "frog", "horse", "ship", "truck"]

TRAINX = "./data/cifar/train_x.npy"
TRAINY = "./data/cifar/train_y.npy"
TESTX = "./data/cifar/test_x.npy"
TESTY = "./data/cifar/test_y.npy"
DATABASEX = "./data/cifar/data_x.npy"
DATABASEY = "./data/cifar/data_y.npy"


def resizeX(X, w, h):
    N = X.shape[0]
    # Resize img to 256 * 256
    resized = np.zeros((N, h, w, 3))
    for i in range(N):
        resized[i] = cv2.resize(X[i], (w, h))
    return resized


def normalize(inp):
    inp /= 255.0
    inp = 2 * inp - 1.0
    return inp


class Cifar(object):
    """docstring for Cifar."""

    def __init__(self, mode, resizeWidth, resizeHeight):
        if mode != "database" and mode != "train" and mode != "query" and mode != "all":
            raise AttributeError("Argument of mode is invalid.")
        self._mode = mode
        self.TrainName = ["data_batch_1", "data_batch_2",
                          "data_batch_3", "data_batch_4", "data_batch_5"]
        self.TestName = ["test_batch"]
        self.DataFolder = "./data/cifar-10-batches-py"
        self._width = resizeWidth
        self._height = resizeHeight
        self.readCifar()
        self._counts = self.X.shape[0]

    def ReadAll(self):
        imgs = list()
        labels = list()

        filenames = [path.join(self.DataFolder, name)
                     for name in (self.TrainName + self.TestName)]
        for f in filenames:
            x, y = LoadCifarFile(f)
            imgs.append(x)
            labels.append(y)

        imgs = np.array(imgs)
        labels = np.array(labels)

        X = imgs.reshape(
            (-1, imgs.shape[2], imgs.shape[3], imgs.shape[4]))
        Y = labels.reshape((-1))

        idx = np.random.permutation(X.shape[0])

        self.X = X[idx]
        self.Y = Y[idx]

    def readCifar(self):
        if self._mode == "all":
            print('all')
            self.ReadAll()

            self.DataNum = self.X.shape[0]
            self.ClassNum = NUM_CLASSES
            self.n_samples = self.DataNum

            self.Onehot()

            print("Loaded from Saved file")
            print("Label shape:", self.Y.shape)
            print("Data shape:", self.X.shape)
            return
        if self._mode == "database":
            print('database')
            self.X = np.load(DATABASEX).astype(np.uint8)
            self.Y = np.load(DATABASEY)

            self.DataNum = self.X.shape[0]
            self.ClassNum = NUM_CLASSES
            self.n_samples = self.DataNum

            self.Onehot()

            print("Loaded from Saved file")
            print("Label shape:", self.Y.shape)
            print("Data shape:", self.X.shape)
            return
        if self._mode == "train":
            print('train')
            self.X = np.load(TRAINX).astype(np.uint8)
            self.Y = np.load(TRAINY)

            self.DataNum = self.X.shape[0]
            self.ClassNum = NUM_CLASSES
            self.n_samples = self.DataNum

            self.Onehot()

            print("Loaded from Saved file")
            print("Label shape:", self.Y.shape)
            print("Data shape:", self.X.shape)
            return
        else:
            print('query')
            self.X = np.load(TESTX).astype(np.uint8)
            self.Y = np.load(TESTY)

            self.DataNum = self.X.shape[0]
            self.ClassNum = NUM_CLASSES
            self.n_samples = self.DataNum

            self.Onehot()

            print("Loaded from Saved file")
            print("Label shape:", self.Y.shape)
            print("Data shape:", self.X.shape)
            return

    def Onehot(self):
        # one-hot encoding
        y = np.zeros((self.X.shape[0], 10), dtype=int)
        y[range(self.X.shape[0]), self.Y] = 1
        self.Y = y

    def Check(self):
        rnd_idx = np.random.permutation(self.X.shape[0])

        i = 0

        import matplotlib.pyplot as plt

        _, axes = plt.subplots(4, 4)

        for ax in axes.ravel():
            ax.imshow(self.X[rnd_idx[i]].astype(int))
            ax.set_title(LABEL_NAME[self.Y[rnd_idx[i]]])
            i += 1
        plt.show()

    # normalize [0~255] to [-1, 1]]

    def Get(self, index):
        return resizeX(self.X[index], self._width, self._height), self.Y[index]

    def GetX(self):
        return resizeX(self.X, self._width, self._height)

    @property
    def SamplesCount(self):
        return self._counts


def LoadCifarFile(filename):
    with open(filename, 'rb') as f:
        if py3:
            datadict = p.load(f, encoding='latin-1')
        else:
            datadict = p.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y
