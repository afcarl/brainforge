from csxdata.utilities.loader import pull_mnist_data
from homyd.datamodel import FlatDataset

from brainforge import LayerStack, BackpropNetwork
from brainforge.layers import DenseLayer
from brainforge.gradientcheck import GradientCheck

ds = FlatDataset.from_multiarrays(*pull_mnist_data(split=0))
ds.split_new_subset_from("learning", "testing", split_ratio=0.2, randomize=True)
ds.set_encoding("onehot")
inshape, outshape = ds.shapes

layers = LayerStack(input_shape=ds.shapes[0], layers=[
    DenseLayer(60, activation="tanh", compiled=False),
    DenseLayer(outshape[0], activation="softmax", compiled=False)
])
ann = BackpropNetwork(layers, cost="cxent", optimizer="adam")

ann.fit(*ds.table("learning", m=5), batch_size=5, epochs=1, verbose=1)
GradientCheck(ann).run(*ds.table("learning", m=5), throw=False)
ann.fit(*ds.table("learning"), batch_size=64, epochs=30, validation=ds.table("testing"))
