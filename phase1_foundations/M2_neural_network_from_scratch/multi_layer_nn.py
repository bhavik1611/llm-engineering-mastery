"""
Multi-layer neural network from scratch (NumPy)
with configurable depth, choice of per-layer activation, and initialization.

Usage:
    layers = [
        (input_dim, h1_dim, "relu"),   # Layer 1: input to h1, ReLU
        (h1_dim, h2_dim, "tanh"),      # Layer 2: h1 to h2, tanh
        (h2_dim, output_dim, "sigmoid") # Out: h2 to output, sigmoid (e.g. for BCE)
    ]
    nn = MultiLayerNN(layers, initialization="he", learning_rate=0.05)
"""

import numpy as np
from layer import Layer


class MultiLayerNN:
    """
    Fully-connected feedforward neural network with arbitrary depth and activation.

    Attributes:
        layers: list of Layer objects
    """

    def __init__(
        self,
        layers,
        **kwargs
    ):
        """
        Args:
            layers: list of (in_dim, out_dim, activation_string), for each layer.
            initialization: "xavier" | "he" | "normal" | "constant" (default "xavier")
            learning_rate: learning rate for gradient descent
            seed: random seed for reproducibility
        """
        seed = kwargs.get('seed', None)
        learning_rate = kwargs.get('learning_rate', 0.1)
        initialization = kwargs.get('initialization', "xavier")
        if seed is not None:
            np.random.seed(seed)

        self.learning_rate = learning_rate
        self.layer_specs = layers  # keep for reference
        self.layers = [
            Layer(in_dim, out_dim, act_name, initialization=initialization)
            for in_dim, out_dim, act_name in layers
        ]
        self._loss_history = []

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input (n_samples, input_dim)
        Returns:
            caches: list of (z, a) for each layer
        """
        a = x
        caches = []
        for layer in self.layers:
            z, a = layer.forward(a)
            caches.append((z, a))
        return caches

    def compute_loss(self, y, y_hat, eps=1e-15):
        """
        BCE loss for output sigmoid/other-prob layer (supports 1D or 2D y).
        Args:
            y: true labels, shape (n,) or (n, output_dim)
            y_hat: predicted, shape (n, output_dim)
        """
        y_hat = np.clip(y_hat, eps, 1 - eps)
        y = np.atleast_2d(y) if y.ndim == 1 else y
        y_hat = np.atleast_2d(y_hat)
        if y.shape[1] != y_hat.shape[1]:
            y = y.reshape(-1, y_hat.shape[1])
        loss = -np.mean(y * np.log(y_hat) + (1. - y) * np.log(1. - y_hat))
        return float(loss)

    def backward(self, x, y, caches):
        """
        Backpropagation for multi-layer NN (arbitrary depth and activations).
        Args:
            x: input, shape (n, input_dim)
            y: true labels, shape (n,) or (n, output_dim)
            caches: output of forward() (list of (z, a))
        Returns:
            grads: list [(dW, db)] per layer, in same order as layers.
        """
        n = x.shape[0]
        wGrads = []
        bGrads = []

        activations = [x] + [a for (_, a) in caches]
        zs = [z for (z, _) in caches]
        y_hat = activations[-1]
        y = np.atleast_2d(y) if y.ndim == 1 else y
        if y.shape[1] != y_hat.shape[1]:
            y = y.reshape(-1, y_hat.shape[1])

        # BCE+sigmoid: dL/dz = y_hat - y. Layer expects dA = dL/da, so dA = (y_hat-y)/(y_hat*(1-y_hat)).
        eps = 1e-15
        y_hat_safe = np.clip(y_hat, eps, 1 - eps)
        dA = (y_hat - y) / (y_hat_safe * (1 - y_hat_safe))
        for layer_idx in reversed(range(len(self.layers))):
            layer = self.layers[layer_idx]
            z = zs[layer_idx]
            a_prev = activations[layer_idx]
            dW, db, dA_prev = layer.backward(dA, z, a_prev, n)
            wGrads.insert(0, dW)
            bGrads.insert(0, db)
            dA = dA_prev

        return wGrads, bGrads

    def update_params(self, wGrads, bGrads):
        for layer, dW, db in zip(self.layers, wGrads, bGrads):
            layer.weights -= self.learning_rate * dW
            layer.biases -= self.learning_rate * db

    def fit(self, x, y, epochs, verbose=False):
        self._loss_history = []
        for epoch in range(epochs):
            caches = self.forward(x)
            y_hat = caches[-1][1]
            loss = self.compute_loss(y, y_hat)
            self._loss_history.append(loss)

            wGrads, bGrads = self.backward(x, y, caches)
            self.update_params(wGrads, bGrads)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        return self._loss_history

    def predict(self, x, threshold=0.5):
        """
        Return binary predictions (default) or output of last layer.
        """
        caches = self.forward(x)
        y_hat = caches[-1][1]
        return (y_hat >= threshold).astype(np.float64)

    def gradient_check(self, x, y, epsilon=1e-5, verbose=True):
        """
        Compares analytical and numerical gradients for sanity check.
        """
        caches = self.forward(x)
        wGrads, bGrads = self.backward(x, y, caches)

        max_rel_error = 0.0

        for layer_idx, layer in enumerate(self.layers):
            param_pairs = [("W", layer.weights, wGrads[layer_idx]), ("b", layer.biases, bGrads[layer_idx])]
            for pname, param, grad_analytic in param_pairs:
                grad_numeric = np.zeros_like(param)
                it = np.nditer(param, flags=["multi_index"])
                while not it.finished:
                    idx = it.multi_index
                    orig_val = param[idx]
                    param[idx] = orig_val + epsilon
                    caches_pos = self.forward(x)
                    loss_pos = self.compute_loss(y, caches_pos[-1][1])
                    param[idx] = orig_val - epsilon
                    caches_neg = self.forward(x)
                    loss_neg = self.compute_loss(y, caches_neg[-1][1])
                    param[idx] = orig_val
                    grad_numeric[idx] = (loss_pos - loss_neg) / (2 * epsilon)
                    it.iternext()
                num_flat = grad_numeric.ravel()
                ana_flat = grad_analytic.ravel()
                rel_error = np.max(np.abs(num_flat - ana_flat) / (np.abs(num_flat) + np.abs(ana_flat) + 1e-12))
                max_rel_error = max(max_rel_error, rel_error)
                if verbose:
                    print(f"Layer {layer_idx} {pname}: max rel error = {rel_error:.2e}")
        if verbose:
            print(f"\nGradient check: max relative error = {max_rel_error:.2e}")
        return max_rel_error
