"""
Two-layer neural network from scratch using NumPy.

Architecture: Input → Linear → Sigmoid → Linear → Sigmoid → BCE Loss

Model:
    z1 = X @ W1 + b1     # (n, input_dim) @ (input_dim, hidden_dim) -> (n, hidden_dim)
    a1 = sigmoid(z1)     # (n, hidden_dim)
    z2 = a1 @ W2 + b2    # (n, hidden_dim) @ (hidden_dim, output_dim) -> (n, output_dim)
    y_hat = sigmoid(z2) # (n, output_dim)

Loss: Binary Cross-Entropy (mean over batch)
"""

import numpy as np


def _sigmoid(z):
    """Numerically stable sigmoid."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _sigmoid_derivative(z):
    """Derivative of sigmoid: sigma'(z) = sigma(z) * (1 - sigma(z))."""
    s = _sigmoid(z)
    return s * (1 - s)


class TwoLayerNN:
    """
    Two-layer feedforward neural network for binary classification.

    Input → Linear(W1, b1) → Sigmoid → Linear(W2, b2) → Sigmoid → BCE

    Attributes:
        W1: weights for first layer, shape (input_dim, hidden_dim)
        b1: bias for first layer, shape (1, hidden_dim)
        W2: weights for second layer, shape (hidden_dim, output_dim)
        b2: bias for second layer, shape (1, output_dim)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        **kwargs
    ):
        """
        Initialize the two-layer network.

        Args:
            input_dim: Number of input features.
            hidden_dim: Number of hidden units.
            output_dim: Number of outputs (default 1 for binary).
            learning_rate: Step size for gradient descent.
            weight_init_std: Std for weight initialization.
            seed: Random seed for reproducibility.
        """
        if kwargs.get('seed', None) is not None:
            np.random.seed(kwargs.get('seed'))

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = kwargs.get('output_dim', 1)
        self.lr = kwargs.get('learning_rate', 0.1)

        # W1: (input_dim, hidden_dim), b1: (1, hidden_dim)
        self.w1 = np.random.randn(input_dim, hidden_dim) * kwargs.get('weight_init_std', 0.01)
        self.b1 = np.zeros((1, hidden_dim))

        # W2: (hidden_dim, output_dim), b2: (1, output_dim)
        self.w2 = np.random.randn(hidden_dim, self.output_dim) * kwargs.get('weight_init_std', 0.01)
        self.b2 = np.zeros((1, self.output_dim))

        self._loss_history = []

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            X: Input features, shape (n_samples, input_dim).

        Returns:
            z1: Pre-activation layer 1, shape (n_samples, hidden_dim).
            a1: Activation layer 1, shape (n_samples, hidden_dim).
            z2: Pre-activation layer 2, shape (n_samples, output_dim).
            y_hat: Output probabilities, shape (n_samples, output_dim).
        """
        # Layer 1: (n, input_dim) @ (input_dim, hidden_dim) + (1, hidden_dim) -> (n, hidden_dim)
        z1 = x @ self.w1 + self.b1
        a1 = _sigmoid(z1)

        # Layer 2: (n, hidden_dim) @ (hidden_dim, output_dim) + (1, output_dim) -> (n, output_dim)
        z2 = a1 @ self.w2 + self.b2
        y_hat = _sigmoid(z2)

        return z1, a1, z2, y_hat

    def compute_loss(self, y, y_hat, eps=1e-15):
        """
        Binary cross-entropy loss with numerical stability.

        L = -mean[y * log(y_hat) + (1-y) * log(1-y_hat)]

        Args:
            y: True labels, shape (n_samples,) or (n_samples, output_dim).
            y_hat: Predicted probabilities, shape (n_samples, output_dim).

        Returns:
            Scalar loss (mean over batch).
        """
        y_hat = np.clip(y_hat, eps, 1 - eps)
        y = np.atleast_2d(y) if y.ndim == 1 else y
        if y.shape[1] != y_hat.shape[1]:
            y = y.reshape(-1, 1)
        # mean over batch
        return -float(np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

    def backward(self, x, y, **kwargs):
        """
        Backward pass: compute gradients w.r.t. all parameters.

        Derivation:
            dL/dz2 = y_hat - y
            dL/dW2 = a1.T @ dL_dz2 / n
            dL/db2 = mean(dL_dz2)
            dL/da1 = dL_dz2 @ W2.T
            dL/dz1 = dL_da1 * sigmoid'(z1)
            dL/dW1 = X.T @ dL_dz1 / n
            dL/db1 = mean(dL_dz1)

        Args:
            X: Input, shape (n, input_dim).
            y: True labels, shape (n,) or (n, output_dim).
            z1, a1, z2: Pre-activations and activations (z2 unused: dL/dz2 = y_hat - y).
            y_hat: Output probabilities from forward pass.

        Returns:
            Dict with keys: W1, b1, W2, b2, each containing gradient array.
        """
        n = x.shape[0]
        z1, z2, a1, y_hat = kwargs.get('z1'), kwargs.get('z2'), kwargs.get('a1'), kwargs.get('y_hat')
        assert z2.shape == y_hat.shape  # consistency check (z2 -> y_hat via sigmoid)
        y = np.atleast_2d(y) if y.ndim == 1 else y
        if y.shape[1] != y_hat.shape[1]:
            y = y.reshape(-1, 1)

        # Output layer gradients
        # dL/dz2: (n, output_dim)
        dL_dz2 = y_hat - y

        # dL/dW2: (hidden_dim, output_dim)
        dL_dW2 = (a1.T @ dL_dz2) / n

        # dL/db2: (1, output_dim)
        dL_db2 = np.mean(dL_dz2, axis=0, keepdims=True)

        # Hidden layer gradients
        # dL/da1: (n, hidden_dim)
        dL_da1 = dL_dz2 @ self.w2.T

        # dL/dz1: (n, hidden_dim)
        dL_dz1 = dL_da1 * _sigmoid_derivative(z1)

        # dL/dW1: (input_dim, hidden_dim)
        dL_dW1 = (x.T @ dL_dz1) / n

        # dL/db1: (1, hidden_dim)
        dL_db1 = np.mean(dL_dz1, axis=0, keepdims=True)

        return {
            "W1": dL_dW1,
            "b1": dL_db1,
            "W2": dL_dW2,
            "b2": dL_db2,
        }

    def update_params(self, grads):
        """
        Update parameters using gradient descent.

        Args:
            grads: Dict from backward() with keys W1, b1, W2, b2.
        """
        self.w1 -= self.lr * grads["W1"]
        self.b1 -= self.lr * grads["b1"]
        self.w2 -= self.lr * grads["W2"]
        self.b2 -= self.lr * grads["b2"]

    def fit(self, x, y, epochs, verbose=False):
        """
        Train the network for a fixed number of epochs.

        Args:
            X: Training features, shape (n_samples, input_dim).
            y: Training labels, shape (n_samples,) or (n_samples, output_dim).
            epochs: Number of training epochs.
            verbose: If True, print loss every epoch.

        Returns:
            loss_history: List of loss values per epoch.
        """
        self._loss_history = []

        for epoch in range(epochs):
            z1, a1, z2, y_hat = self.forward(x)
            loss = self.compute_loss(y, y_hat)
            self._loss_history.append(loss)

            grads = self.backward(x, y, **{
                'z1': z1, 'a1': a1, 'z2': z2, 'y_hat': y_hat
            })
            self.update_params(grads)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

        return self._loss_history

    def predict(self, x, threshold=0.5):
        """
        Predict binary class labels.

        Args:
            X: Input features, shape (n_samples, input_dim).
            threshold: Decision threshold (default 0.5).

        Returns:
            Binary predictions, shape (n_samples, output_dim), values in {0, 1}.
        """
        _, _, _, y_hat = self.forward(x)
        return (y_hat >= threshold).astype(np.float64)

    def gradient_check(self, x, y, epsilon=1e-5):
        """
        Verify analytical gradients against numerical (finite difference) gradients.

        For each parameter theta: numerical_grad ≈ (L(theta+eps) - L(theta-eps)) / (2*eps)
        Reports max relative error across all parameters.

        Args:
            x: Input, shape (n, input_dim).
            y: Labels, shape (n,) or (n, output_dim).
            epsilon: Step size for finite difference.

        Returns:
            max_rel_error: Maximum relative error across all parameters.
        """
        z1, a1, z2, y_hat = self.forward(x)
        grads = self.backward(x, y, **{
            'z1': z1, 'a1': a1, 'z2': z2, 'y_hat': y_hat
        })

        param_list = [
            ("W1", self.w1), ("b1", self.b1), ("W2", self.w2), ("b2", self.b2)
        ]
        max_rel_error = 0.0

        for name, param in param_list:
            numerical = np.zeros_like(param)
            analytical = grads[name]

            it = np.nditer(param, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index

                # L(theta + epsilon)
                param[idx] += epsilon
                _, _, _, y_hat_plus = self.forward(x)
                loss_plus = self.compute_loss(y, y_hat_plus)

                # L(theta - epsilon)
                param[idx] -= 2 * epsilon
                _, _, _, y_hat_minus = self.forward(x)
                loss_minus = self.compute_loss(y, y_hat_minus)

                # Restore
                param[idx] += epsilon

                numerical[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                it.iternext()

            # Relative error
            denom = np.abs(analytical) + np.abs(numerical) + 1e-12
            rel_error = np.max(np.abs(analytical - numerical) / denom)
            max_rel_error = max(max_rel_error, rel_error)
            print(f"  {name}: max rel error = {rel_error:.2e}")

        print(f"\nGradient check: max relative error = {max_rel_error:.2e}")
        return max_rel_error
