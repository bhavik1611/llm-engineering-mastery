"""
Optimisers for neural network training (NumPy).

Provides SGD, Momentum, RMSProp, and Adam for use with MultiLayerNN.
Each optimizer maintains per-parameter state and updates parameters in-place.

Usage:
    from optimisers import SGD, Momentum, RMSProp, Adam
    nn = MultiLayerNN([2, 10, 1], optimizer=Adam(lr=0.01), ...)
    nn.fit(X, y, epochs=100)
"""

import numpy as np


class Optimizer:
    """Base class for parameter optimizers."""

    def step(self, layers, wGrads, bGrads):
        """
        Update parameters given gradients.

        Args:
            layers: list of Layer objects (each has .weights, .biases)
            wGrads: list of weight gradients (same length as layers)
            bGrads: list of bias gradients
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Vanilla stochastic gradient descent.
    Update: θ ← θ - lr * ∇L
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers, wGrads, bGrads):
        for layer, dW, db in zip(layers, wGrads, bGrads):
            layer.weights -= self.lr * dW
            layer.biases -= self.lr * db


class Momentum(Optimizer):
    """
    SGD with momentum.
    v ← β*v + ∇L
    θ ← θ - lr*v
    """

    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self._m_W = None
        self._m_b = None

    def _ensure_state(self, layers):
        if self._m_W is None:
            self._m_W = [np.zeros_like(l.weights) for l in layers]
            self._m_b = [np.zeros_like(l.biases) for l in layers]

    def step(self, layers, wGrads, bGrads):
        self._ensure_state(layers)
        for i, (layer, dW, db) in enumerate(zip(layers, wGrads, bGrads)):
            self._m_W[i] = self.beta * self._m_W[i] + dW
            self._m_b[i] = self.beta * self._m_b[i] + db
            layer.weights -= self.lr * self._m_W[i]
            layer.biases -= self.lr * self._m_b[i]


class RMSProp(Optimizer):
    """
    RMSProp: per-parameter adaptive learning rate.
    g² ← ρ*g² + (1-ρ)*∇L²
    θ ← θ - lr * ∇L / (√g² + ε)
    """

    def __init__(self, lr=0.01, rho=0.9, eps=1e-8):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self._g2_W = None
        self._g2_b = None

    def _ensure_state(self, layers):
        if self._g2_W is None:
            self._g2_W = [np.zeros_like(l.weights) for l in layers]
            self._g2_b = [np.zeros_like(l.biases) for l in layers]

    def step(self, layers, wGrads, bGrads):
        self._ensure_state(layers)
        for i, (layer, dW, db) in enumerate(zip(layers, wGrads, bGrads)):
            self._g2_W[i] = self.rho * self._g2_W[i] + (1 - self.rho) * (dW ** 2)
            self._g2_b[i] = self.rho * self._g2_b[i] + (1 - self.rho) * (db ** 2)
            layer.weights -= self.lr * dW / (np.sqrt(self._g2_W[i]) + self.eps)
            layer.biases -= self.lr * db / (np.sqrt(self._g2_b[i]) + self.eps)


class Adam(Optimizer):
    """
    Adam: combines momentum and RMSProp with bias correction.
    m ← β1*m + (1-β1)*∇L
    v ← β2*v + (1-β2)*∇L²
    m̂ ← m / (1 - β1ᵗ)
    v̂ ← v / (1 - β2ᵗ)
    θ ← θ - lr * m̂ / (√v̂ + ε)
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m_W = None
        self._m_b = None
        self._v_W = None
        self._v_b = None
        self._t = 0

    def _ensure_state(self, layers):
        if self._m_W is None:
            self._m_W = [np.zeros_like(l.weights) for l in layers]
            self._m_b = [np.zeros_like(l.biases) for l in layers]
            self._v_W = [np.zeros_like(l.weights) for l in layers]
            self._v_b = [np.zeros_like(l.biases) for l in layers]

    def step(self, layers, wGrads, bGrads):
        self._ensure_state(layers)
        self._t += 1
        for i, (layer, dW, db) in enumerate(zip(layers, wGrads, bGrads)):
            self._m_W[i] = self.beta1 * self._m_W[i] + (1 - self.beta1) * dW
            self._m_b[i] = self.beta1 * self._m_b[i] + (1 - self.beta1) * db
            self._v_W[i] = self.beta2 * self._v_W[i] + (1 - self.beta2) * (dW ** 2)
            self._v_b[i] = self.beta2 * self._v_b[i] + (1 - self.beta2) * (db ** 2)
            mW_hat = self._m_W[i] / (1 - self.beta1 ** self._t)
            mb_hat = self._m_b[i] / (1 - self.beta1 ** self._t)
            vW_hat = self._v_W[i] / (1 - self.beta2 ** self._t)
            vb_hat = self._v_b[i] / (1 - self.beta2 ** self._t)
            layer.weights -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            layer.biases -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


def get_optimizer(name, **kwargs):
    """
    Factory: return optimizer by name.

    Args:
        name: "sgd" | "momentum" | "rmsprop" | "adam"
        **kwargs: passed to optimizer constructor (e.g. lr, beta)
    """
    name = name.lower()
    if name == "sgd":
        return SGD(**kwargs)
    if name == "momentum":
        return Momentum(**kwargs)
    if name == "rmsprop":
        return RMSProp(**kwargs)
    if name == "adam":
        return Adam(**kwargs)
    raise ValueError(f"Unknown optimizer: {name}. Use sgd, momentum, rmsprop, or adam.")
