from ._llops import dense_forward, dense_backward
from brainforge.util.typing import zX


class DenseOp:

    @staticmethod
    def forward(X, W, b=None):
        if b is None:
            b = zX(W.shape[-1])
        return dense_forward(X, W, b)

    @staticmethod
    def backward(X, E, W):
        gradient = dense_backward(X, E, W)
        nW = gradient[:W.size].reshape(W.shape)
        nb = gradient[W.size:W.shape[-1]]
        dX = gradient[-X.size:].reshape(X.shape)
        return nW, nb, dX
