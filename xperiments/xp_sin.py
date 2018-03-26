import numpy as np

from matplotlib import pyplot as plt

from brainforge import DifferentialNeuroevolution
from brainforge.layers import DenseLayer
from brainforge.evolution import DifferentialEvolution

# from keras.models import Sequential
# from keras.layers import Dense

# np.random.seed(1234)

rX = np.linspace(-6., 6., 200)[:, None]
rY = np.sin(rX)

arg = np.arange(len(rX))
np.random.shuffle(arg)
targ, varg = arg[:100], arg[100:]
targ.sort()
varg.sort()

tX, tY = rX[targ], rY[targ]
vX, vY = rX[varg], rY[varg]

tX += np.random.randn(*tX.shape) / np.sqrt(tX.size*0.25)


def forge_net():
    net = DifferentialNeuroevolution(input_shape=(1,), layerstack=[
        DenseLayer(30, activation="relu"),
        DenseLayer(30, activation="relu"),
        DenseLayer(1, activation="linear")
    ], cost="mse", fitness_func=fitness)
    return net


def fitness(ind, net, x, y):
    net.set_weights(ind)
    cost = net.learn_batch(x, y)
    return net.get_weights, cost


def mate(ind1, ind2):
    return np.mean((ind1, ind2), axis=0) + forge_net().get_weights(unfold=True)


def xperiment():
    net, pop = evolution()
    tpred = net.predict(tX)
    vpred = net.predict(vX)
    plt.ion()
    plt.plot(tX, tY, "b--", alpha=0.5, label="Training data (noisy)")
    plt.plot(rX, rY, "r--", alpha=0.5, label="Validation data (clean)")
    plt.ylim(-2, 2)
    plt.plot(rX, np.ones_like(rX), c="black", linestyle="--")
    plt.plot(rX, -np.ones_like(rX), c="black", linestyle="--")
    plt.plot(rX, np.zeros_like(rX), c="grey", linestyle="--")
    tobj, = plt.plot(tX, tpred, "bo", markersize=3, alpha=0.5, label="Training pred")
    vobj, = plt.plot(vX, vpred, "ro", markersize=3, alpha=0.5, label="Validation pred")
    templ = "Batch: {:>5} Cost = {:.4f}"
    t = plt.title(templ.format(0, 0))
    plt.legend()
    batchno = 1
    while 1:
        pop.run(epochs=1, survival_rate=0.5, mutation_rate=0.1, verbosity=0, x=tX, y=tY, net=net)
        net.set_weights(pop.best)
        # cost = net.train_on_batch(tX, tY)
        tpred = net.predict(tX)
        vpred = net.predict(vX)
        tobj.set_data(tX, tpred)
        vobj.set_data(vX, vpred)
        plt.pause(0.01)
        t.set_text(templ.format(batchno, pop.grades.min()))
        batchno += 1


if __name__ == '__main__':
    xperiment()
