#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function, division
import os
from os import path
import numpy as np
import pickle as p
import cv2
from Utils import ProgressBar
import json

import sys

py3 = sys.version_info >= (3, 4)

NUM_CLASSES = 100
DATABASE_PATH = './data/imagenet/database.txt'
TRAIN_PATH = './data/imagenet/train.txt'
TEST_PATH = './data/imagenet/query.txt'
CHOOSED_PATH = './data/imagenet/choosed.json'

class Imagenet(object):
    """docstring for Imagenet."""

    def __init__(self, mode, resizeWidth, resizeHeight):
        if (mode != "database" and mode != "train" and mode != "query" and mode != "all"):
            raise AttributeError("Argument of mode is invalid.")
        self._mode = mode
        self._width = resizeWidth
        self._height = resizeHeight
        self.readPath()


    def ReadAll(self):
        for i in range(self.n_samples):
            this_lint = self.lines[i].strip().split()
            self._img[i] = cv2.resize(cv2.imread(this_lint[0]), (256, 256))
            cl = int(this_lint[1])
            self._label[i] = [0] * self._classnum
            self._label[i][self._choosed.index(cl)] = 1
            self._load[i] = 1
            self._load_num += 1
            if self._load_num % 500 == 0:
                print(self._load_num / self.n_samples)

        if self._load_num == self.n_samples:
            self._status = 1
            self.X = np.array(self._img)
            self.Y = np.array(self._label)
            print('All images read')
            print("X:")
            print(self.X.shape)
            print("Y:")
            print(self.Y.shape)


    def readPath(self):
        with open(CHOOSED_PATH, 'r') as fp:
            self._choosed = json.load(fp)['class']
            self._classnum = len(self._choosed)
            assert self._classnum == 100
        if self._mode == "all":
            print('all ** not implement **')
            return
        elif self._mode == "database":
            print('database')
            self.lines = open(DATABASE_PATH, 'r').readlines()
        elif self._mode == "train":
            print('train')
            self.lines = open(TRAIN_PATH, 'r').readlines()
        else:
            print('query')
            self.lines = open(TEST_PATH, 'r').readlines()

        
        print("total lines: %d" % len(self.lines))

        self.DataNum = len(self.lines)
        self.ClassNum = NUM_CLASSES
        self.n_samples = self.DataNum
        self._counts = self.n_samples
    
        self._img = [0] * self.n_samples
        self._label = [0] * self.n_samples
        self._load = [0] * self.n_samples
        self._load_num = 0
        self._status = 0
        return


    def resizeX(self, X, w, h):
        N = X.shape[0]
        # Resize img to 256 * 256
        resized = np.zeros((N, h, w, 3))
        for i in range(N):
            resized[i] = cv2.resize(X[i], (w, h))
        return resized

    # normalize [0~255] to [-1, 1]]
    def normalize(self, inp):
        inp /= 255.0
        inp = 2 * inp - 1.0
        return inp

    def Get(self, index):
        if self._status:
            return self.resizeX(self.X[index], self._width, self._height), self.Y[index]
        else:
            ret_img = []
            ret_label = []
            for i in index:
                if i >= self.DataNum:
                    break
                try:
                    if not self._load[i]:
                        this_lint = self.lines[i].strip().split()
                        self._img[i] = cv2.resize(cv2.imread(this_lint[0]), (256, 256))
                        cl = int(this_lint[1])
                        self._label[i] = [0] * self._classnum
                        self._label[i][self._choosed.index(cl)] = 1
                        self._load[i] = 1
                        self._load_num += 1
                    ret_img.append(self._img[i])
                    ret_label.append(self._label[i])
                except:
                    print('cannot open', self.lines[i])
                # else:
                    # print(self.lines[i])

            if self._load_num == self.n_samples:
                self._status = 1
                self.X = np.array(self._img)
                self.Y = np.array(self._label)
                print('All images read')
                print("X:")
                print(self.X.shape)
                print("Y:")
                print(self.Y.shape)
            return self.resizeX(np.asarray(ret_img), self._width, self._height), np.asarray(ret_label)

    def GetX(self):
        self.ReadAll()
        return self.resizeX(self.X, self._width, self._height)

    @property
    def SamplesCount(self):
        return self._counts