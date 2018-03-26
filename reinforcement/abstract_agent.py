import abc

import numpy as np

from .agentconfig import AgentConfig
from .experience import xp_factory


class AgentBase(abc.ABC):

    type = ""

    def __init__(self, network, nactions, agentconfig: AgentConfig, **kw):
        if agentconfig is None:
            agentconfig = AgentConfig(**kw)
        self.nactions = nactions
        self.net = network
        self.shadow_net = network.layers.get_weights()
        self.xp = xp_factory(agentconfig.xpsize, "drop", agentconfig.time)
        self.cfg = agentconfig

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, state, reward):
        raise NotImplementedError

    @abc.abstractmethod
    def accumulate(self, state, reward):
        raise NotImplementedError

    def learn_batch(self):
        X, Y = self.xp.replay(self.cfg.bsize)
        N = len(X)
        if N < self.cfg.bsize:
            return 0.
        costs = self.net.fit(X, Y, verbose=0, epochs=1)
        # return np.mean(cost.history["loss"])
        return np.mean(costs)

    def push_weights(self):
        W = self.net.layers.get_weights(unfold=True)
        D = np.linalg.norm(self.shadow_net - W)
        self.shadow_net *= (1. - self.cfg.tau)
        self.shadow_net += self.cfg.tau * self.net.layers.get_weights(unfold=True)
        return D / len(W)

    def pull_weights(self):
        self.net.layers.set_weights(self.shadow_net, fold=True)
