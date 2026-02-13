import numpy as np
import matplotlib.pyplot as plt

def generate_linearly_separable(n_per_class=50):
    """Generate two blobs that can be separated by a line."""
    # Class 0: bottom-left blob
    X0 = np.random.randn(n_per_class, 2) + np.array([-2, -2])
    y0 = np.zeros(n_per_class)
    
    # Class 1: top-right blob
    X1 = np.random.randn(n_per_class, 2) + np.array([2, 2])
    y1 = np.ones(n_per_class)
    
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    return X, y

def plot_x_y(X, y):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='#e74c3c', label='Class 0', edgecolors='black', linewidths=0.5)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='#3498db', label='Class 1', edgecolors='black', linewidths=0.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Linearly Separable 2D Dataset')
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

# Train/test split (80/20)
def train_test_split(X, y, test_frac=0.2, seed=42):
    np.random.seed(seed)
    n = len(y)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_frac))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def mse_loss(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def bce_loss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
