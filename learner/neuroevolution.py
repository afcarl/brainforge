from .abstract_learner import Learner
from .backpropagation import BackpropNetwork
from ..evolution import DifferentialEvolution, MemeticAlgorithm


class NeuroEvolution(Learner):

    def __init__(self, layerstack, cost, population_size, name="", **kw):
        super().__init__(layerstack, cost, name, **kw)
        self.fitness_function = kw.pop("fitness_function", self.fitness)
        self.fitness_weights = kw.pop("fitness_weights", [1.])
        self.on_accuracy = kw.pop("on_accuracy", False)
        self.population_size = population_size
        self.kws = kw
        self.population = None

    def learn_batch(self, X, Y, **kw):
        args = {"epochs": 1, "survival_rate": 0.8, "mutation_rate": 0.1, "verbosity": 0}
        args.update(kw)
        self.population.run(**kw, X=X, Y=Y)
        self.layers.set_weights(self.population.best)
        return self.population.grade.min()

    def fitness(self, genome, X, Y):
        raise NotImplementedError


class DifferentialNeuroevolution(NeuroEvolution):

    def __init__(self, layerstack, cost="mse", population_size=100, name="", **kw):
        super().__init__(layerstack, cost, population_size, name, **kw)
        self.population = DifferentialEvolution(
            loci=self.layers.nparams,
            fitness_function=self.fitness_function,
            fitness_weights=self.fitness_weights,
            limit=self.population_size, **kw
        )

    def fitness(self, genome, X, Y):
        self.layers.set_weights(self.as_weights(genome))
        result = self.evaluate(X, Y, classify=self.on_accuracy)
        return (1. - result[-1]) if self.on_accuracy else result


class MemeticNeuroevolution(NeuroEvolution):

    def __init__(self, layerstack, cost="mse", population_size=100, name="", **kw):
        super().__init__(layerstack, cost, population_size, name, **kw)
        self.ann = BackpropNetwork(layerstack, cost, optimizer="sgd")
        self.population = MemeticAlgorithm(
            loci=self.ann.nparams,
            fitness_function=ff,
            fitness_weights=fw,
            limit=population_size, **kw
        )

    def fitness(self, genome, X, Y):
        self.ann.set_weights(genome, fold=False)
        self.ann.learn_batch(X, Y)
        result = self.evaluate(X, Y, classify=self.on_accuracy)
        return self.ann.get_weights(unfold=True), (1. - result[-1]) if self.on_accuracy else result
