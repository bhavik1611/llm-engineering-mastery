"""
Activation functions for neural networks.
"""
from abc import abstractmethod
import numpy as np

class Activation:
    """
    Base class for neural network activation functions.

    Provides an interface for activation functions.
    Subclasses should implement the `_apply` and `_gradient` methods.

    Methods
    -------
    __call__(z):
        Applies the activation to the input.
    apply(z):
        Applies the activation function to input `z` (with documentation).
    gradient(z):
        Computes the derivative of the activation w.r.t. `z` (with doc).
    """
    def __call__(self, z):
        return self.apply(z)

    @abstractmethod
    def apply(self, z):
        """
        Applies the activation function to input `z`.

        Args:
            z: Input to the activation function.

        Returns:
            The activated output.
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, z):
        """
        Computes the derivative of the activation function with respect to `z`.

        Args:
            z: Input to the activation function.

        Returns:
            The derivative of the activation function with respect to `z`.
        """
        raise NotImplementedError

class Sigmoid(Activation):
    """
    Sigmoid activation function: squashes input values to the (0, 1) interval.
    """
    def apply(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def gradient(self, z):
        s = self.apply(z)
        return s * (1 - s)

class ReLU(Activation):
    """
    ReLU (Rectified Linear Unit) activation function.
    Outputs zero for negative inputs, and the input itself for positive inputs.
    """
    def apply(self, z):
        return np.maximum(0, z)

    def gradient(self, z):
        return (z > 0).astype(z.dtype)

class Tanh(Activation):
    """
    Tanh (hyperbolic tangent) activation function: squashes input values to (-1, 1).
    """
    def apply(self, z):
        return np.tanh(z)

    def gradient(self, z):
        t = np.tanh(z)
        return 1 - t**2

class ActivationFactory:
    """
    Factory class for creating activation function instances.

    Supported activations:
      - "sigmoid"
      - "relu"
      - "tanh"

    Raises
    ------
    ValueError
        If the requested activation is not recognized.
    """
    _activations = {
        "sigmoid": Sigmoid,
        "relu": ReLU,
        "tanh": Tanh
    }

    @classmethod
    def get_activation(cls, method):
        """
        Get an activation function instance by method name.

        Parameters
        ----------
        method : str
            Name of the activation function (e.g., "sigmoid", "relu", "tanh").

        Returns
        -------
        Activation
            An instance of a subclass of Activation corresponding to `method`.

        Raises
        ------
        ValueError
            If the method is not found in the supported activations.
        """
        try:
            return cls._activations[method.lower()]()
        except KeyError:
            raise ValueError(
                f"Unknown activation: {method!r}. "
                f"Available: {list(cls._activations.keys())}"
        )
