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
from layer import Layer


class TwoLayerNN:
    """
    Two-layer feedforward neural network for binary classification.

    Input → Layer1 → Sigmoid → Layer2 → Sigmoid → BCE

    Attributes:
        layer1: First fully-connected layer (input → hidden)
        layer2: Second fully-connected layer (hidden → output)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
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
        seed = kwargs.get('seed', None)
        if seed is not None:
            np.random.seed(seed)

        self.lr = kwargs.get('learning_rate', 0.1)
        weight_init_std = kwargs.get('weight_init_std', 0.01)

        self.layer1 = Layer(
            input_dim,
            hidden_dim,
            "sigmoid",
            weight_init_std=weight_init_std,
        )
        self.layer2 = Layer(
            hidden_dim,
            output_dim,
            "sigmoid",
            weight_init_std=weight_init_std,
        )

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
        z1, a1 = self.layer1.forward(x)
        z2, y_hat = self.layer2.forward(a1)
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
        return -float(np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

    def backward(self, x, y, **kwargs):
        """
        Backward pass: compute gradients w.r.t. all parameters.

        Args:
            X: Input, shape (n, input_dim).
            y: True labels, shape (n,) or (n, output_dim).
            z1, a1, z2, y_hat: From forward pass.

        Returns:
            Dict with keys: W1, b1, W2, b2, each containing gradient array.
        """
        n = x.shape[0]
        z1, z2, a1, y_hat = kwargs.get('z1'), kwargs.get('z2'), kwargs.get('a1'), kwargs.get('y_hat')
        assert z2.shape == y_hat.shape
        y = np.atleast_2d(y) if y.ndim == 1 else y
        if y.shape[1] != y_hat.shape[1]:
            y = y.reshape(-1, 1)

        # BCE+sigmoid: dL/dz2 = y_hat - y. Layer expects dA = dL/da, so dA = (y_hat-y)/(y_hat*(1-y_hat)).
        eps = 1e-15
        y_hat_safe = np.clip(y_hat, eps, 1 - eps)
        dA2 = (y_hat - y) / (y_hat_safe * (1 - y_hat_safe))
        dW2, db2, dA1 = self.layer2.backward(dA2, z2, a1, n)
        dW1, db1, _ = self.layer1.backward(dA1, z1, x, n)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def update_params(self, grads):
        """
        Update parameters using gradient descent.

        Args:
            grads: Dict from backward() with keys W1, b1, W2, b2.
        """
        self.layer1.weights -= self.lr * grads["W1"]
        self.layer1.biases -= self.lr * grads["b1"]
        self.layer2.weights -= self.lr * grads["W2"]
        self.layer2.biases -= self.lr * grads["b2"]

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

            grads = self.backward(x, y, z1=z1, a1=a1, z2=z2, y_hat=y_hat)
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
        """
        z1, a1, z2, y_hat = self.forward(x)
        grads = self.backward(x, y, z1=z1, a1=a1, z2=z2, y_hat=y_hat)

        param_list = [
            ("W1", self.layer1.weights), ("b1", self.layer1.biases),
            ("W2", self.layer2.weights), ("b2", self.layer2.biases),
        ]
        max_rel_error = 0.0

        for name, param in param_list:
            numerical = np.zeros_like(param)
            analytical = grads[name]

            it = np.nditer(param, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index

                param[idx] += epsilon
                _, _, _, y_hat_plus = self.forward(x)
                loss_plus = self.compute_loss(y, y_hat_plus)

                param[idx] -= 2 * epsilon
                _, _, _, y_hat_minus = self.forward(x)
                loss_minus = self.compute_loss(y, y_hat_minus)

                param[idx] += epsilon

                numerical[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                it.iternext()

            denom = np.abs(analytical) + np.abs(numerical) + 1e-12
            rel_error = np.max(np.abs(analytical - numerical) / denom)
            max_rel_error = max(max_rel_error, rel_error)
            print(f"  {name}: max rel error = {rel_error:.2e}")

        print(f"\nGradient check: max relative error = {max_rel_error:.2e}")
        return max_rel_error
