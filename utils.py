import torch
import numpy as np
import imageio

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
    # Initialize the lambda-returns with the bootstrap value.
    v_lambda = next_values[:, -1] * (1.0 - terminals[:, -1])
    horizon = next_values.shape[1]
    lambda_values = np.empty_like(next_values)

    for t in reversed(range(horizon)):
        td = (
            rewards[:, t]
            + (1.0 - terminals[:, t]) * (1.0 - lambda_) * discount * next_values[:, t]
        )
        v_lambda = td + v_lambda * lambda_ * discount
        lambda_values = lambda_values.at[:, t].set(v_lambda)
    return lambda_values


def discount(factor, length):
    # Create a discount array for a given horizon length.
    # This is used to scale rewards over multiple steps.
    d = np.cumprod(factor * np.ones((length - 1,)))
    d = np.concatenate([np.ones((1,)), d])
    return d


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


class ObsNormalizer:
    def __init__(self, shape, eps=1e-8):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 0
        self.eps = eps

    def update(self, x):
        self.count += 1
        if self.count == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean += (x - old_mean) / self.count
            self.var += (x - old_mean) * (x - self.mean)

    def normalize(self, x):
        std = np.sqrt(self.var / (self.count + self.eps))
        return (x - self.mean) / (std + self.eps)


# Symlog functions
def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# Gumbel-Softmax function for discrete latent variables
def gumbel_softmax(logits, tau=1.0, hard=False):
    gumbels = -torch.empty_like(logits).exponential_().log()
    y = (logits + gumbels) / tau
    y_soft = torch.softmax(y, dim=-1)

    if hard:
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return (y_hard - y_soft).detach() + y_soft
    else:
        return y_soft


def save_animation(frames, filename):
    with imageio.get_writer(filename, mode="I", loop=0) as writer:
        for frame in frames:
            writer.append_data(frame)
