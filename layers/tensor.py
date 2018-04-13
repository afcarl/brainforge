from .abstract_layer import LayerBase, NoParamMixin
from ..util import zX, zX_like, white


class PoolLayer(NoParamMixin, LayerBase):

    def __init__(self, fdim, compiled=True):
        LayerBase.__init__(self, activation="linear", trainable=False)
        if compiled:
            print("Compiling PoolLayer...")
            from ..llatomic.lltensor_op import MaxPoolOp
        else:
            from ..atomic import MaxPoolOp
        self.fdim = fdim
        self.filter = None
        self.op = MaxPoolOp()

    def connect(self, brain):
        ic, iy, ix = brain.outshape[-3:]
        if any((iy % self.fdim, ix % self.fdim)):
            raise RuntimeError(f"Incompatible shapes: {ix} % {iy} & {self.fdim}")
        super().connect(brain)

    def feedforward(self, questions):
        """
        Implementation of a max pooling layer.

        :param questions: numpy.ndarray, a batch of outsize from the previous layer
        :return: numpy.ndarray, max pooled batch
        """
        self.output, self.filter = self.op.forward(questions, self.fdim)
        return self.output

    def backpropagate(self, delta):
        """
        Calculates the error of the previous layer.
        :param delta:
        :return: numpy.ndarray, the errors of the previous layer
        """
        return self.op.backward(delta, self.filter)

    @property
    def outshape(self):
        ic, iy, ix = self.inshape
        return ic, iy // self.fdim, ix // self.fdim

    def __str__(self):
        return "Pool-{}x{}".format(self.fdim, self.fdim)


class ConvLayer(LayerBase):

    def __init__(self, nfilters, filterx=3, filtery=3, **kw):
        super().__init__(activation=kw.get("activation", "linear"), **kw)
        self.nfilters = nfilters
        self.fx = filterx
        self.fy = filtery
        self.depth = 0
        self.stride = 1
        self.inshape = None
        self.op = None

    def connect(self, brain):
        if self.compiled:
            print("Compiling ConvLayer...")
            from ..llatomic import ConvolutionOp
        else:
            from ..atomic import ConvolutionOp
        c, iy, ix = brain.outshape[-3:]
        if any((iy < self.fy, ix < self.fx)):
            raise RuntimeError(f"Incompatible shapes: iy ({iy}) < fy ({self.fy}) OR ix ({ix}) < fx ({self.fx})")
        super().connect(brain)
        self.op = ConvolutionOp()
        self.weights = white(self.nfilters, c, self.fx, self.fy)
        self.biases = zX(self.nfilters)
        self.nabla_b = zX_like(self.biases)
        self.nabla_w = zX_like(self.weights)

    def feedforward(self, X):
        self.inputs = X
        self.output = self.activation.forward(self.op.forward(X, self.weights, "valid"))
        return self.output

    def backpropagate(self, delta):
        delta *= self.activation.backward(self.output)
        self.nabla_w = self.op.forward(
            self.inputs.transpose(1, 0, 2, 3),
            delta.transpose(1, 0, 2, 3),
            mode="valid"
        ).transpose(1, 0, 2, 3)
        # self.nabla_b = error.sum()  # TODO: why is this commented out???
        rW = self.weights[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)
        return self.op.forward(delta, rW, "full")

    @property
    def outshape(self):
        iy, ix = self.inshape[-2:]
        return self.nfilters, iy - self.fy, ix - self.fx

    def __str__(self):
        return "Conv({}x{}x{})-{}".format(self.nfilters, self.fy, self.fx, str(self.activation)[:4])
