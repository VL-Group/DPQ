#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import numpy as np
from os import path

SIFT10K_PATH = './data/SIFT10K/'
SIFT1M_PATH = './data/SIFT1M/'


def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def ivecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def sift10k_read():
    if not path.exists(SIFT10K_PATH):
        raise Exception("SIFT not found")
    base = fvecs_read(SIFT10K_PATH + 'sift_base.fvecs')
    learn = fvecs_read(SIFT10K_PATH + 'sift_learn.fvecs')
    query = fvecs_read(SIFT10K_PATH + 'sift_query.fvecs')
    gt = ivecs_read(SIFT10K_PATH + 'sift_groundtruth.ivecs')
    print(base.shape, learn.shape, query.shape)
    return base, learn, query, gt


def sift1m_read():
    if not path.exists(SIFT1M_PATH):
        raise Exception("SIFT not found")
    base = fvecs_read(SIFT1M_PATH + 'sift_base.fvecs')
    learn = fvecs_read(SIFT1M_PATH + 'sift_learn.fvecs')
    query = fvecs_read(SIFT1M_PATH + 'sift_query.fvecs')
    gt = ivecs_read(SIFT1M_PATH + 'sift_groundtruth.ivecs')
    print(base.shape, learn.shape, query.shape, gt.shape)
    return base, learn, query, gt


class Sift(object):
    """docstring for Sift."""

    def __init__(self, mode):
        print(mode)
        if (mode != "database" and mode != "train" and mode != "test" and mode != "all"):
            raise AttributeError("Argument of mode is invalid.")
        self._mode = mode
        base, learn, query, gt = sift1m_read()
        if mode == "database":
            self.X = base
        elif mode == "train":
            self.X = base
        elif mode == "test":
            self.X = query
        else:
            self.X = base

        self.GroundTruth = gt
        self.base = base
        self.learn = learn
        self.query = query
        self._counts = self.X.shape[0]

    def Get(self, index):
        return self.X[index]

    @property
    def SamplesCount(self):
        return self._counts
