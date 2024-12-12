import torch
import numpy as np
import yaml
from types import SimpleNamespace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(image):
    # Preprocess raw images by normalizing pixel values from [0, 255] to [-0.5, 0.5].
    return image / 255.0 - 0.5


def quantize(image):
    # Inverse of preprocess: map [-0.5, 0.5] back to [0, 255] as uint8.
    return ((image + 0.5) * 255).astype(np.uint8)


def compute_lambda_values(
    next_values,
    rewards,
    terminals,
    discount,
    lambda_,
):
    """
    Compute lambda-returns for imagined trajectories:
    The lambda-returns combine bootstrap values with multi-step returns,
    providing a trade-off between bias and variance.

    next_values: Value estimates for each step (including the last bootstrap).
    rewards: Immediate rewards at each step.
    terminals: Terminal signals (1 if episode ends).
    discount: Discount factor for future rewards.
    lambda_: Parameter controlling the weighting between 1-step returns and bootstrap values.
    """
    # print("next_values shape:", next_values.shape)
    # print("rewards shape:", rewards.shape)
    # print("terminals shape:", terminals.shape)


    # Initialize the lambda-returns with the bootstrap value.
    v_lambda = next_values[:, -1] * (1.0 - terminals[:, -1])
    horizon = next_values.shape[1]
    lambda_values = torch.empty_like(next_values)

    for t in reversed(range(horizon)):
        td = (
            rewards[:, t]
            + (1.0 - terminals[:, t]) * (1.0 - lambda_) * discount * next_values[:, t]
        )
        v_lambda = td + v_lambda * lambda_ * discount
        lambda_values[:, t] = v_lambda
    return lambda_values


def discount(factor, length):
    # Create a discount array for a given horizon length.
    # This is used to scale rewards over multiple steps.
    d = np.cumprod(factor * np.ones((length - 1,)))
    d = np.concatenate([np.ones((1,)), d])
    return torch.tensor(d)


def global_norm(grads):
    """
    Compute the global L2 norm of a set of gradients in PyTorch.

    Args:
        grads (iterable): An iterable of gradient tensors.

    Returns:
        float: The global L2 norm.
    """
    total_norm = 0.0
    for grad in grads:
        if grad is not None:  # Skip parameters with no gradient
            param_norm = grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


class Config:
    @staticmethod
    def _dict_to_namespace(d):
        """
        Recursively convert a dictionary to a SimpleNamespace.
        """
        if isinstance(d, dict):
            return SimpleNamespace(
                **{k: Config._dict_to_namespace(v) for k, v in d.items()}
            )
        elif isinstance(d, list):
            return [Config._dict_to_namespace(v) for v in d]
        else:
            return d

    @staticmethod
    def load_from_yaml(filepath):
        with open(filepath, "r") as file:
            config = yaml.safe_load(file)
        # Convert nested config dictionary to namespace
        return Config._dict_to_namespace(config.get("defaults", {}))
