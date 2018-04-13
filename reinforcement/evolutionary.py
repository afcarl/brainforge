import numpy as np

from .abstract_agent import AgentBase, AgentConfig

from brainforge import BackpropNetwork
from evolute import DifferentialEvolution
from evolute.fitness import SimpleFunction
from evolute.operators import Elitism, UniformLocuswiseMutation


class EvolutionaryStrategies(AgentBase):

    def __init__(self, network: BackpropNetwork, nactions, agentconfig: AgentConfig, **kw):
        super().__init__(network, nactions, agentconfig, **kw)
        self.population = DifferentialEvolution(
            loci=network.nparams,
            fitness_wrapper=SimpleFunction(network.cost),
            limit=100, selection_op=Elitism(selection_rate=0.98),
            mutate_op=UniformLocuswiseMutation(rate=1., low=-3., high=3.)
        )
        self.champion = np.zeros(network.nparams)
        self.reward_running = 0.

    def reset(self):
        pass

    def sample(self, state, reward):
        self.reward_running += reward
        return self.net.predict(state)

    def accumulate(self, state, reward):
        return self.net.predict(state)
