"""
Multi-layer neural network from scratch (NumPy)
with configurable depth, choice of per-layer activation, and initialization.

Usage (per-layer activation):
    layers = [
        (input_dim, h1_dim, "relu"),   # Layer 1: input to h1, ReLU
        (h1_dim, h2_dim, "tanh"),      # Layer 2: h1 to h2, tanh
        (h2_dim, output_dim, "sigmoid") # Out: h2 to output, sigmoid (e.g. for BCE)
    ]
    nn = MultiLayerNN(layers, initialization="he", learning_rate=0.05)

Usage (layer_dims + single activation):
    nn = MultiLayerNN([2, 4, 4, 1], activation="sigmoid", initialization="xavier")
    nn = MultiLayerNN([2, 4, 1], activation="relu", weight_init_std=1e-6)
"""

import numpy as np
from layer import Layer


def _is_layer_specs(obj):
    """True if obj is list of (in_dim, out_dim, act_name) tuples."""
    if not obj or not isinstance(obj, (list, tuple)):
        return False
    first = obj[0]
    return isinstance(first, (list, tuple)) and len(first) == 3


def _compute_activation_stats(caches, layer_specs):
    """Compute activation statistics per layer from caches and layer specs."""
    stats = {}
    for i, (z, a) in enumerate(caches):
        act_name = layer_specs[i][2].lower() if layer_specs else "sigmoid"
        if act_name == "sigmoid":
            saturation_pct = np.mean((a < 0.01) | (a > 0.99)) * 100
            dead_pct = 0.0
        else:
            saturation_pct = np.mean(a <= 0) * 100
            dead_pct = np.mean(a <= 0) * 100 if act_name == "relu" else 0.0
        stats[f"layer_{i}"] = {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "saturation_pct": float(saturation_pct),
            "dead_pct": float(dead_pct),
        }
    return stats


class MultiLayerNN:
    """
    Fully-connected feedforward neural network with arbitrary depth and activation.

    Attributes:
        layers: list of Layer objects
        layer_specs: list of (in_dim, out_dim, activation_string)
    """

    def __init__(self, layers_or_dims, **kwargs):
        """
        Args:
            layers_or_dims: Either:
                - list of (in_dim, out_dim, activation_string) for each layer, or
                - list of ints [input_dim, h1, h2, ..., output_dim] (requires activation kwarg)
            activation: (when using layer_dims) "sigmoid" | "relu" | "tanh" for all layers
            initialization: "xavier" | "he" | "normal" | "constant" | "small" | "large"
            weight_init_std: if set, overrides initialization with this scale
            learning_rate: learning rate for gradient descent
            seed: random seed for reproducibility
        """
        seed = kwargs.get("seed", None)
        learning_rate = kwargs.get("learning_rate", 0.1)
        initialization = kwargs.get("initialization", "xavier")
        weight_init_std = kwargs.get("weight_init_std", None)
        if seed is not None:
            np.random.seed(seed)

        if _is_layer_specs(layers_or_dims):
            layer_specs = list(layers_or_dims)
        else:
            dims = list(layers_or_dims)
            if len(dims) < 2:
                raise ValueError("layer_dims must have at least 2 elements (input_dim, output_dim)")
            activation = kwargs.get("activation", "sigmoid")
            layer_specs = [
                (dims[i], dims[i + 1], activation)
                for i in range(len(dims) - 1)
            ]

        self.learning_rate = learning_rate
        self.layer_specs = layer_specs
        if not layer_specs:
            raise ValueError("At least one layer required (e.g. layer_dims with >= 2 dims)")
        self._layer_dims = [s[0] for s in layer_specs] + [layer_specs[-1][1]]

        layer_kw = {"initialization": initialization}
        if weight_init_std is not None:
            layer_kw["weight_init_std"] = weight_init_std

        self.layers = [
            Layer(in_dim, out_dim, act_name, **layer_kw)
            for in_dim, out_dim, act_name in layer_specs
        ]
        self._loss_history = []
        self._grad_norms_history = []
        self._activation_stats_history = []

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

    @property
    def weights(self):
        """List of weight matrices for direct access (e.g. for loss landscape viz)."""
        return [layer.weights for layer in self.layers]

    @property
    def biases(self):
        """List of bias vectors for direct access."""
        return [layer.biases for layer in self.layers]

    @property
    def loss_history(self):
        """Loss per epoch (alias for _loss_history)."""
        return self._loss_history

    @property
    def grad_norms_history(self):
        """Gradient norms per layer per epoch (when track_stats=True in fit)."""
        return self._grad_norms_history

    @property
    def activation_stats_history(self):
        """Activation statistics per layer per epoch (when track_stats=True in fit)."""
        return self._activation_stats_history

    def backward(self, x, y, caches, return_grad_norms=False):
        """
        Backpropagation for multi-layer NN (arbitrary depth and activations).
        Args:
            x: input, shape (n, input_dim)
            y: true labels, shape (n,) or (n, output_dim)
            caches: output of forward() (list of (z, a))
            return_grad_norms: if True, compute and return gradient norms per layer
        Returns:
            wGrads, bGrads, grad_norms: gradients per layer; grad_norms is None if return_grad_norms=False
        """
        n = x.shape[0]
        wGrads = []
        bGrads = []
        grad_norms = [] if return_grad_norms else None

        activations = [x] + [a for (_, a) in caches]
        zs = [z for (z, _) in caches]
        y_hat = activations[-1]
        y = np.atleast_2d(y) if y.ndim == 1 else y
        if y.shape[1] != y_hat.shape[1]:
            y = y.reshape(-1, y_hat.shape[1])

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
            if return_grad_norms:
                grad_norms.insert(
                    0, float(np.sqrt(np.sum(dW ** 2) + np.sum(db ** 2)))
                )
            dA = dA_prev

        return wGrads, bGrads, grad_norms

    def update_params(self, wGrads, bGrads):
        """
        Update the parameters of the network.

        Args:
            wGrads: list of weight gradients
            bGrads: list of bias gradients
        """
        for layer, dW, db in zip(self.layers, wGrads, bGrads):
            layer.weights -= self.learning_rate * dW
            layer.biases -= self.learning_rate * db

    def fit(self, x, y, epochs, **kwargs):
        """
        Train the network.

        Args:
            x: input features
            y: true labels (1D or 2D)
            epochs: number of training epochs
            verbose: if True, print progress every epoch
            track_stats: if True, record grad_norms_history and activation_stats_history
        Returns:
            loss_history
        """
        verbose = kwargs.get("verbose", False)
        track_stats = kwargs.get("track_stats", False)
        self._loss_history = []
        self._grad_norms_history = []
        self._activation_stats_history = []

        for epoch in range(epochs):
            caches = self.forward(x)
            y_hat = caches[-1][1]
            loss = self.compute_loss(y, y_hat)
            self._loss_history.append(loss)

            if track_stats:
                wGrads, bGrads, grad_norms = self.backward(
                    x, y, caches, return_grad_norms=True
                )
                self._grad_norms_history.append(grad_norms)
                stats = _compute_activation_stats(caches, self.layer_specs)
                self._activation_stats_history.append(stats)
            else:
                wGrads, bGrads, _ = self.backward(x, y, caches)

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
        wGrads, bGrads, _ = self.backward(x, y, caches)

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
