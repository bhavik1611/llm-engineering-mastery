"""
Fully-connected layer: linear transform (W, b) + activation.

Used by both TwoLayerNN and MultiLayerNN.
"""

import numpy as np
from activations import ActivationFactory


class Layer:
    """
    Single fully-connected layer: z = a_prev @ W + b, a = activation(z).

    Attributes:
        W: weight matrix, shape (in_dim, out_dim)
        b: bias vector, shape (1, out_dim)
        activation: Activation instance (callable)
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        activation_name,
        **kwargs
    ):
        """
        Args:
            in_dim: Input dimension (fan_in).
            out_dim: Output dimension (fan_out).
            activation_name: Name of activation ("sigmoid", "relu", "tanh").
            initialization: "xavier" | "he" | "normal" | "constant" (default "xavier").
            weight_init_std: overrides initialization with simple scale.
            seed: Random seed for weight initialization.
        """
        initialization = kwargs.get('initialization', "xavier")
        weight_init_std = kwargs.get('weight_init_std', None)
        seed = kwargs.get('seed', None)

        if seed is not None:
            np.random.seed(seed)

        if weight_init_std is not None:
            scale = float(weight_init_std)
        elif initialization == "he":
            scale = np.sqrt(2.0 / in_dim)
        elif initialization == "xavier":
            scale = np.sqrt(1.0 / in_dim)
        elif initialization == "normal":
            scale = 1.0
        elif initialization == "constant":
            scale = 0.01
        else:
            scale = 0.01

        self.weights = np.random.randn(in_dim, out_dim) * scale
        self.biases = np.zeros((1, out_dim))

        act = ActivationFactory.get_activation(activation_name)
        self.activation = act

    def forward(self, a_prev):
        """
        Forward pass: z = a_prev @ W + b, a = activation(z).

        Args:
            a_prev: Input activations, shape (n_samples, in_dim).

        Returns:
            z: Pre-activation, shape (n_samples, out_dim).
            a: Post-activation, shape (n_samples, out_dim).
        """
        z = a_prev @ self.weights + self.biases
        a = self.activation(z)
        return z, a

    def backward(self, dA, z, a_prev, n):
        """
        Backward pass: compute gradients given dL/da (gradient flowing into this layer's output).

        Args:
            dA: Gradient of loss w.r.t. this layer's output a, shape (n_samples, out_dim).
            z: Pre-activation from forward pass.
            a_prev: Input activations from forward pass.
            n: Batch size (for averaging gradients).

        Returns:
            dW: Gradient w.r.t. W, shape (in_dim, out_dim).
            db: Gradient w.r.t. b, shape (1, out_dim).
            dA_prev: Gradient to pass to previous layer, shape (n_samples, in_dim).
        """
        dZ = dA * self.activation.gradient(z)
        dW = (a_prev.T @ dZ) / n
        db = np.mean(dZ, axis=0, keepdims=True)
        dA_prev = dZ @ self.weights.T
        return dW, db, dA_prev
