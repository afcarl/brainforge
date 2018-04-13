"""Wrappers for vector-operations and other functions"""
import numpy as np

from ..util.typing import scalX

s0 = scalX(0.)


class DenseOp:

    @staticmethod
    def forward(X, W, b):
        return np.dot(X, W) + b

    @staticmethod
    def backward(X, E, W):
        nW = np.dot(X.T, E)
        nb = np.sum(E, axis=0)
        dX = np.dot(E, W.T)
        return nW, nb, dX


class ReshapeOp:

    type = "Reshape"

    @staticmethod
    def forward(X, outshape):
        return X.reshape(X.shape[0], *outshape)

    @staticmethod
    def backward(E, inshape):
        return ReshapeOp.forward(E, inshape)
