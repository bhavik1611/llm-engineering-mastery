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

    def linear_forward(self, x):
        """
        Compute the linear transformation (xw + b).

        Args:
            x (np.ndarray): Input data of shape (n_samples, n_features)

        Returns:
            np.ndarray: Linear output
        """
        return np.dot(x, self.weights) + self.bias

    def forward(self, x):
        """
        Full forward pass: linear + sigmoid activation.

        Args:
            x (np.ndarray): Input data

        Returns:
            tuple: (z, y_pred)
                z (np.ndarray): Linear output
                y_pred (np.ndarray): Sigmoid output
        """
        z = self.linear_forward(x)
        y_pred = self.sigmoid(z)
        return z, y_pred

    def gradient(self, x, y, y_pred):
        """
        Compute gradients of loss w.r.t. weights and bias.

        Args:
            x (np.ndarray): Input features
            y (np.ndarray): True labels
            y_pred (np.ndarray): Predictions after sigmoid

        Returns:
            tuple: (gradient_w, gradient_b)
        """
        gradient_w = np.dot(x.T, (y_pred - y)) / len(y) + self.lambda_ * self.weights / len(y)
        gradient_b = np.mean(y_pred - y)
        return gradient_w, gradient_b

    def update_weights(self, x, y, y_pred):
        """
        Perform one step of gradient descent to update weights and bias.

        Args:
            x (np.ndarray): Input features
            y (np.ndarray): True labels
            y_pred (np.ndarray): Predictions after sigmoid
        """
        gradient_w, gradient_b = self.gradient(x, y, y_pred)
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
        l2_loss = 0.5 * self.lambda_ * np.sum(self.weights ** 2) / len(y)
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

    def train(self, x, y):
        """
        Fit the logistic regression model using gradient descent.

        Args:
            x (np.ndarray): Training features (n_samples, n_features)
            y (np.ndarray): Training labels (n_samples,)

        Returns:
            tuple: (weights, bias, losses, grad_norms, history)
        """
        _, n_features = x.shape
        self.weights, self.bias = self.init_weights_and_bias(n_features)
        for _ in range(self.epochs):
            _, y_pred = self.forward(x)
            loss = self.loss(y, y_pred, loss_type='bce')
            self.update_weights(x, y, y_pred)
            self.append_training_history(loss)
            if self.verbose:
                print(f"Epoch {_ + 1}, Loss: {loss}")

        return self.weights, self.bias, self.losses, self.grad_norms, self.history

    def append_training_history(self, loss):
        """
        Record the weights, bias, and loss after each epoch.

        Args:
            loss (float): Loss value for current epoch
        """
        self.history.append((self.weights.copy(), self.bias))
        self.losses.append(loss)

    def predict(self, x):
        """
        Predict binary class labels for input data using the trained model.

        Args:
            x (np.ndarray): Input data

        Returns:
            np.ndarray: Rounded predictions (0 or 1)
        """
        _, y_pred = self.forward(x)
        return np.round(y_pred)

    def evaluate(self, x, y):
        """
        Compute the classification accuracy of the model.

        Args:
            x (np.ndarray): Input features
            y (np.ndarray): True labels

        Returns:
            float: Ratio of correct predictions to total samples
        """
        return np.mean(self.predict(x) == y)
