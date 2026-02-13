'''
Logistic regression model from scratch.
'''

import numpy as np
from utils import bce_loss, mse_loss

class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch with gradient descent.
    Supports L2 regularization, tracks training history.
    """

    def __init__(self, **kwargs):
        """
        Initialize the logistic regression model.

        Args:
            lr (float): Learning rate. Default: 0.01.
            epochs (int): Number of training epochs. Default: 100.
            lambda_ (float): L2 regularization strength. Default: 0.
            verbose (bool): If True, print progress during training. Default: False.
            n_features (int): Number of input features. Default: 2.
        """
        self.lr = kwargs.get('lr', 0.01)
        self.epochs = kwargs.get('epochs', 100)
        self.lambda_ = kwargs.get('lambda_', 0)
        self.verbose = kwargs.get('verbose', False)
        self.weights, self.bias = self.init_weights_and_bias(kwargs.get('n_features', 2))
        self.init_training_history()

    def init_training_history(self):
        """
        Initializes the lists used to keep track of training history:
        losses, gradient norms, and weights/bias per epoch.
        """
        self.losses = []
        self.grad_norms = []
        self.history = []

    def sigmoid(self, z):
        """
        Numerically-stable sigmoid function.

        Args:
            z (np.ndarray): Linear model output

        Returns:
            np.ndarray: Output after applying the sigmoid function
        """
        z = np.clip(z, -500, 500)  # Avoid overflow
        return 1.0 / (1.0 + np.exp(-z))

    def linear_forward(self, X):
        """
        Compute the linear transformation (Xw + b).

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)

        Returns:
            np.ndarray: Linear output
        """
        return np.dot(X, self.weights) + self.bias

    def forward(self, X, w=None, b=None):
        """
        Full forward pass: linear + sigmoid activation.

        Args:
            X (np.ndarray): Input data
            w (np.ndarray, optional): Optionally override weights
            b (float, optional): Optionally override bias

        Returns:
            tuple: (z, y_pred)
                z (np.ndarray): Linear output
                y_pred (np.ndarray): Sigmoid output
        """
        if w is not None and b is not None:
            self.weights = w
            self.bias = b
        z = self.linear_forward(X)
        y_pred = self.sigmoid(z)
        return z, y_pred

    def gradient(self, X, y, y_pred):
        """
        Compute gradients of loss w.r.t. weights and bias.

        Args:
            X (np.ndarray): Input features
            y (np.ndarray): True labels
            y_pred (np.ndarray): Predictions after sigmoid

        Returns:
            tuple: (gradient_w, gradient_b)
        """
        gradient_w = np.dot(X.T, (y_pred - y)) / len(y) + self.lambda_ * self.weights
        gradient_b = np.mean(y_pred - y)
        return gradient_w, gradient_b

    def update_weights(self, X, y, y_pred):
        """
        Perform one step of gradient descent to update weights and bias.

        Args:
            X (np.ndarray): Input features
            y (np.ndarray): True labels
            y_pred (np.ndarray): Predictions after sigmoid
        """
        gradient_w, gradient_b = self.gradient(X, y, y_pred)
        self.weights -= self.lr * gradient_w
        self.bias -= self.lr * gradient_b
        self.grad_norms.append(np.sqrt(np.sum(gradient_w**2) + gradient_b**2))

    def loss(self, y, y_pred, loss_type='bce'):
        """
        Compute the objective function (BCE or MSE) and regularization.

        Args:
            y (np.ndarray): True labels
            y_pred (np.ndarray): Predictions
            loss_type (str): Type of loss ('bce' or 'mse')

        Returns:
            float: Computed loss value
        """
        l2_loss = 0.5 * self.lambda_ * np.sum(self.weights ** 2)
        if loss_type == 'bce':
            loss = bce_loss(y, y_pred) + l2_loss
        elif loss_type == 'mse':
            loss = mse_loss(y, y_pred)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
        return loss

    def init_weights_and_bias(self, n_features):
        """
        Randomly initialize weights and bias for the model.

        Args:
            n_features (int): Number of features

        Returns:
            tuple: (weights, bias)
        """
        self.weights = np.random.randn(n_features) * 0.5
        self.bias = 0.0
        return self.weights, self.bias

    def train(self, X, y):
        """
        Fit the logistic regression model using gradient descent.

        Args:
            X (np.ndarray): Training features (n_samples, n_features)
            y (np.ndarray): Training labels (n_samples,)

        Returns:
            tuple: (weights, bias, losses, grad_norms, history)
        """
        _, n_features = X.shape
        self.weights, self.bias = self.init_weights_and_bias(n_features)
        for _ in range(self.epochs):
            z, y_pred = self.forward(X)
            loss = self.loss(y, y_pred, loss_type='bce')
            self.update_weights(X, y, y_pred)
            self.append_training_history(loss)
            self.verbose and print(f"Epoch {_ + 1}, Loss: {loss}")

        return self.weights, self.bias, self.losses, self.grad_norms, self.history

    def append_training_history(self, loss):
        """
        Record the weights, bias, and loss after each epoch.

        Args:
            loss (float): Loss value for current epoch
        """
        self.history.append((self.weights.copy(), self.bias))
        self.losses.append(loss)

    def predict(self, X):
        """
        Predict binary class labels for input data using the trained model.

        Args:
            X (np.ndarray): Input data

        Returns:
            np.ndarray: Rounded predictions (0 or 1)
        """
        z, y_pred = self.forward(X)
        return np.round(y_pred)

    def evaluate(self, X, y):
        """
        Compute the classification accuracy of the model.

        Args:
            X (np.ndarray): Input features
            y (np.ndarray): True labels

        Returns:
            float: Ratio of correct predictions to total samples
        """
        return np.mean(self.predict(X) == y)