'''
Logistic regression model from scratch.
'''

import numpy as np
from utils import bce_loss, mse_loss

class LogisticRegression:
    def __init__(self, **kwargs):
        self.lr = kwargs.get('lr', 0.01)
        self.epochs = kwargs.get('epochs', 100)
        self.lambda_ = kwargs.get('lambda_', 0)
        self.verbose = kwargs.get('verbose', False)
        self.weights, self.bias = self.init_weights_and_bias(kwargs.get('n_features', 2))
        self.init_training_history()

    def init_training_history(self):
        self.losses = []
        self.grad_norms = []
        self.history = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  # Avoid overflow
        return 1.0 / (1.0 + np.exp(-z))

    def linear_forward(self, X):
        return np.dot(X, self.weights) + self.bias

    def forward(self, X, w=None, b=None):
        if w is not None and b is not None:
            self.weights = w
            self.bias = b
        z = self.linear_forward(X)
        y_pred = self.sigmoid(z)
        return z, y_pred

    def gradient(self, X, y, y_pred):
        gradient_w = np.dot(X.T, (y_pred - y)) / len(y) + self.lambda_ * self.weights
        gradient_b = np.mean(y_pred - y)
        return gradient_w, gradient_b

    def update_weights(self, X, y, y_pred):
        gradient_w, gradient_b = self.gradient(X, y, y_pred)
        self.weights -= self.lr * gradient_w
        self.bias -= self.lr * gradient_b
        self.grad_norms.append(np.sqrt(np.sum(gradient_w**2) + gradient_b**2))

    def loss(self, y, y_pred, loss_type='bce'):
        l2_loss = 0.5 * self.lambda_ * np.sum(self.weights ** 2)
        if loss_type == 'bce':
            loss = bce_loss(y, y_pred) + l2_loss
        elif loss_type == 'mse':
            loss = mse_loss(y, y_pred)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
        return loss

    def init_weights_and_bias(self, n_features):
        self.weights = np.random.randn(n_features) * 0.5
        self.bias = 0.0
        return self.weights, self.bias

    def train(self, X, y):
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
        self.history.append((self.weights.copy(), self.bias))
        self.losses.append(loss)

    def predict(self, X):
        z, y_pred = self.forward(X)
        return np.round(y_pred)

    def evaluate(self, X, y):
        return np.mean(self.predict(X) == y)