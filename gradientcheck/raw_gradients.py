import numpy as np


def analytical_gradients(gcobj, X, Y):
    network = gcobj.net
    print("Calculating analytical gradients...")
    print("Forward pass:", end=" ")
    preds = network.predict(X)
    print("done! Backward pass:", end=" ")
    delta = network.cost.derivative(outputs=preds, targets=Y)
    network.backpropagate(delta)
    print("done!")
    return network.get_gradients(unfold=True)


def numerical_gradients(gcobj, X, Y):
    network = gcobj.net
    ws = network.get_weights(unfold=True)
    numgrads = np.zeros_like(ws)
    perturb = np.zeros_like(ws)

    nparams = ws.size
    print("Calculating numerical gradients...")
    for i in range(nparams):
        print(f"\r{nparams} / {i+1}", end=" ")
        perturb[i] += gcobj.eps

        network.set_weights(ws + perturb, fold=True)
        pred1 = network.predict(X)
        cost1 = network.cost(pred1, Y)
        network.set_weights(ws - perturb, fold=True)
        pred2 = network.predict(X)
        cost2 = network.cost(pred2, Y)

        numgrads[i] = (cost1 - cost2)
        perturb[i] = 0.

    numgrads /= (2. * gcobj.eps)
    network.set_weights(ws, fold=True)

    print("Done!")

    return numgrads
