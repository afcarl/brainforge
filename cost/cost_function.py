import numpy as np

from ..util.typing import scalX

s0 = scalX(0.)
s05 = scalX(0.5)
s1 = scalX(1.)
s2 = scalX(2.)


class CostFunction:

    type = ""

    def __call__(self, outputs, targets): pass

    def __str__(self): return self.type if self.type else self.__class__.__name__.lower()

    @staticmethod
    def derivative(outputs, targets):
        return outputs - targets


class _MeanSquaredError(CostFunction):
    type = "mse"

    def __call__(self, outputs, targets):
        return s05 * np.linalg.norm(outputs - targets) ** s2


class _CategoricalCrossEntropy(CostFunction):
    type = "cxent"

    def __call__(self, outputs, targets):
        return -(targets * np.log(outputs)).sum()

    @staticmethod
    def ugly_derivative(outputs, targets):
        enum = targets - outputs
        denom = (outputs - s1) * outputs
        return enum / denom


class _BinaryCrossEntropy(CostFunction):
    type = "bxent"

    def __call__(self, outputs, targets):
        return -(targets * np.log(outputs) + (s1 - targets) * np.log(s1 - outputs)).sum()


class _Hinge(CostFunction):
    type = "hinge"

    def __call__(self, outputs, targets):
        return (np.maximum(s0, s1 - targets * outputs)).sum()

    @staticmethod
    def derivative(outputs, targets):
        """Using subderivatives, d/da = -y whenever output > 0"""
        out = -targets
        out[outputs > s1] = s0
        return out


cost_functions = dict(
    bxent=_BinaryCrossEntropy(),
    cxent=_CategoricalCrossEntropy(),
    mse=_MeanSquaredError(),
    hinge=_Hinge())

cost_functions.update(dict(
    xent=cost_functions["cxent"],
    mean_squared_error=cost_functions["mse"],
    categorical_crossentropy=cost_functions["cxent"],
    binary_crossentropy=cost_functions["bxent"],
))


def cost_factory(name, output_activation=None):
    name = name.lower()
    fun = cost_functions[name]
    if output_activation is not None:
        if str(fun) == "cxent" and str(output_activation) != "softmax":
            raise ValueError("Categorical cross entropy must be used with softmax activation!")
        if str(fun) == "bxent" and str(output_activation) != "sigmoid":
            raise ValueError("Binary cross entropy must be used with sigmoid activation!")
    return fun
