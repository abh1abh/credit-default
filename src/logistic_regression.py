import numpy as np
import pandas as pd


def sigmoid(x):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-x))


def compute_loss(y_true, y_pred):
    """Compute the binary cross-entropy loss."""
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def logistic_regression_fit(
    X: pd.DataFrame, y: pd.Series, learning_rate=0.01, n_iterations=1000
):
    """Fit the logistic regression model using gradient descent."""
    n_samples, n_features = X.shape

    weights: np.ndarray = np.zeros(n_features)
    bias: float = 0
    loss: list = []
    for _ in range(n_iterations):
        linear_model = np.dot(X, weights) + bias
        y_predicted = sigmoid(linear_model)
        loss.append(compute_loss(y, y_predicted))
        # Compute gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        db = (1 / n_samples) * np.sum(y_predicted - y)

        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias, loss


def predict(X: pd.DataFrame, weights: np.ndarray, bias: float):
    """Make predictions using the logistic regression model."""
    linear_model = np.dot(X, weights) + bias
    return sigmoid(linear_model)


def compute_scaling_params(X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute the mean and standard deviation for each feature."""
    mean = X.mean()
    std = X.std()
    if (std == 0).any():
        raise ValueError(
            "Standard deviation is 0 for one or more features. Check your data for constant columns."
        )
    return mean, std


def scale_features(X: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    """Scale the features using the provided mean and standard deviation."""
    return (X - mean) / std
