from evolute import DifferentialEvolution
from evolute.evaluation import SimpleFitness

from .abstract_learner import Learner


class NeuroEvolution(Learner):

    def __init__(self, layerstack, cost, population_size=100, name="", **kw):
        super().__init__(layerstack, cost, name, **kw)
        self.on_accuracy = kw.pop("on_accuracy", False)
        self.population_size = population_size
        self.population = None
        self.population = DifferentialEvolution(
            loci=self.layers.nparams,
            fitness_wrapper=SimpleFitness(self.fitness),
            limit=self.population_size
        )

    def learn_batch(self, X, Y, **kw):
        args = {"epochs": 1, "survival_rate": 0.2, "mutation_rate": 0.5, "verbosity": 0}
        args.update(kw)
        self.population.epoch(**kw, X=X, Y=Y)
        self.layers.set_weights(self.population.get_best())
        return self.population.fitnesses.min()

    def fitness(self, genome, X, Y):
        self.layers.set_weights(genome)
        result = self.evaluate(X, Y, classify=self.on_accuracy)
        return (1. - result[-1]) if self.on_accuracy else result
