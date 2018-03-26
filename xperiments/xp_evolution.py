import numpy as np
from matplotlib import pyplot as plt

from brainforge.evolution import DifferentialEvolution


def upscale(ind: np.ndarray):
    x = ind * 10.
    return x


def fitness(ind):
    return np.linalg.norm(TARGET - upscale(ind)),


def matefn1(ind1, ind2):
    return np.where(np.random.uniform() < 0.5, ind1, ind2)


def matefn2(ind1, ind2):
    return np.add(ind1, ind2) / 2.


TARGET = np.array([3., 3.])


pop = DifferentialEvolution(
    loci=2,
    fitness_function=fitness,
    mate_function=matefn1,
    limit=100)

plt.ion()
obj = plt.plot(*upscale(pop.individuals.T), "bo", markersize=2)[0]
plt.xlim([-2, 11])
plt.ylim([-2, 11])

X, Y = np.linspace(-2, 11, 50), np.linspace(-2, 11, 50)
X, Y = np.meshgrid(X, Y)
Z = np.array([fitness(np.array([x, y])/10.) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
CS = plt.contour(X, Y, Z, cmap="hot")
plt.clabel(CS, inline=1, fontsize=10)
plt.show()
means, stds, bests = [], [], []
for i in range(30):
    m, s, b = pop.run(1, verbosity=0, mutation_rate=0.1)
    means += m
    stds += s
    bests += b
    obj.set_data(*upscale(pop.individuals.T))
    plt.pause(0.1)

means, stds, bests = tuple(map(np.array, (means, stds, bests)))
plt.close()
plt.ioff()
Xs = np.arange(1, len(means) + 1)
plt.plot(Xs, means, "b-")
plt.plot(Xs, means+stds, "g--")
plt.plot(Xs, means-stds, "g--")
plt.plot(Xs, bests, "r-")
plt.xlim([Xs.min()-1, Xs.max()+1])
plt.ylim([bests.min()-1, (means+stds).max()+1])
plt.show()
