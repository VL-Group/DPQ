#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function, division
import numpy as np

from Cifar import Cifar
from NUS_21 import NUS_21
from Imagenet import Imagenet



class Dataset(object):
    """docstring for Dataset."""

    def __init__(self, dataset, mode, batchSize, W, H):
        self.mode = mode
        print(dataset)
        dataset = dataset.upper()
        mode = mode.lower()
        if dataset == "CIFAR":
            self.data = Cifar(mode, W, H)
        elif dataset == "IMAGENET":
            self.data = Imagenet(mode, W, H)
        elif dataset == "NUS":
            self.data = NUS_21(mode, W, H)
        else:
            raise NameError("No datset named {0}".format(dataset))
        self._current = 0
        self._batchSize = batchSize
        if self.mode == "train":
            self.choice = np.random.permutation(self.data.SamplesCount)
        else:
            self.choice = np.arange(0, self.data.SamplesCount, 1)

    def NextBatch(self):
        idx = self.choice[self._current: (self._current + self._batchSize)]
        self._index = idx
        self._current += self._batchSize
        # print("[{0}/{1}]".format(self._current, self.data.SamplesCount))
        return self.data.Get(idx)

    def Index(self, i, batchSize):
        idx = np.arange(i*batchSize, (i+1)*batchSize, 1)
        return self.data.Get(idx)

    @property
    def EpochComplete(self):
        complete = (self._current + self._batchSize) > self.data.SamplesCount
        if complete:
            self._current = 0
            if self.mode == "train":
                self.choice = np.random.permutation(self.data.SamplesCount)
        return complete

    @property
    def Progress(self):
        return self._current / self.data.SamplesCount

    @staticmethod
    def PreparetoEval(setName, W, H):
        setName = setName.upper()
        print(setName)
        if setName == "NUS":
            database = NUS_21('database', W, H)
            query = NUS_21('query', W, H)
        elif setName == "CIFAR":
            database = Cifar("database", W, H)
            query = Cifar("query", W, H)
        elif setName == 'IMAGENET':
            database = Imagenet('database', W, H)
            query = Imagenet('query', W, H)
        queryX = query.GetX()
        queryY = query.Y
        return queryX, queryY, database
