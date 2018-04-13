import abc

import numpy as np

from brainforge.model.layerstack import LayerStack
from brainforge.cost import cost_factory, CostFunction
from brainforge.util import batch_stream


class Learner:

    def __init__(self, layerstack, cost="mse", name="", **kw):
        if not isinstance(layerstack, LayerStack):
            if "input_shape" not in kw:
                raise RuntimeError("Please supply input_shape as a keyword argument!")
            layerstack = LayerStack(kw["input_shape"], layers=layerstack)
        self.layers = layerstack
        self.cost = cost
        if not isinstance(self.cost, CostFunction):
            self.cost = cost_factory(self.cost, output_activation=layerstack[-1].activation)
        self.name = name
        self.age = 0

    def fit_generator(self, generator, lessons_per_epoch, epochs=30, classify=True, validation=None, verbose=1, **kw):
        epcosts = []
        for epoch in range(1, epochs+1):
            if verbose:
                print(f"Epoch {epochs}/{epoch}")
            epcosts += self.epoch(generator, no_lessons=lessons_per_epoch, classify=classify,
                                  validation=validation, verbose=verbose, **kw)
        return epcosts

    def fit(self, X, Y, batch_size=20, epochs=30, classify=True, validation=None, verbose=1, shuffle=True, **kw):
        lkw = {k: v for k, v in locals().items() if k != "self"}
        datastream = batch_stream(lkw.pop("X"), lkw.pop("Y"), m=lkw.pop("batch_size"), shuffle=lkw.pop("shuffle"))
        return self.fit_generator(datastream, len(X), **lkw, **kw)

    def epoch(self, generator, no_lessons, classify=True, validation=None, verbose=1, **kw):
        costs = []
        done = 0

        self.layers.learning = True
        while done < no_lessons:
            batch = next(generator)
            cost = self.learn_batch(*batch, **kw)
            costs.append(cost)

            done += len(batch[0])
            if verbose:
                print("\rDone: {0:>6.1%} Cost: {1: .5f}\t "
                      .format(done/no_lessons, np.mean(costs)), end="")
        self.layers.learning = False
        if verbose:
            print("\rDone: {0:>6.1%} Cost: {1: .5f}\t ".format(1., np.mean(costs)), end="")
            if validation is not None:
                self._print_progress(validation, classify)
            print()

        self.age += no_lessons
        return costs

    def _print_progress(self, validation, classify):
        results = self.evaluate(*validation, classify=classify)

        chain = "Testing cost: {0:.5f}"
        if classify:
            tcost, tacc = results
            accchain = " accuracy: {0:.2%}".format(tacc)
        else:
            tcost = results
            accchain = ""
        print(chain.format(tcost) + accchain, end="")

    def predict(self, X):
        return self.layers.feedforward(X)

    def evaluate(self, X, Y, batch_size=32, classify=True, shuffle=False, verbose=False):
        N = X.shape[0]
        batches = batch_stream(X, Y, m=batch_size, shuffle=shuffle, infinite=False)

        cost, acc = [], []
        for bno, (x, y) in enumerate(batches, start=1):
            if verbose:
                print("\rEvaluating: {:>7.2%}".format((bno*batch_size) / N), end="")
            pred = self.predict(x)
            cost.append(self.cost(pred, y) / len(x))
            if classify:
                pred_classes = np.argmax(pred, axis=1)
                trgt_classes = np.argmax(y, axis=1)
                eq = np.equal(pred_classes, trgt_classes)
                acc.append(eq.mean())
        results = np.mean(cost)
        if classify:
            results = (results, np.mean(acc))
        return results

    @abc.abstractmethod
    def learn_batch(self, X, Y, **kw):
        raise NotImplementedError

    @property
    def output(self):
        return self.layers[-1].output

    @property
    def nparams(self):
        return self.layers.nparams
