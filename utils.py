import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, act_dim):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.act_dim = act_dim
        self.obs_buffer = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.act_buffer = np.zeros(capacity, dtype=np.int64)
        self.rew_buffer = np.zeros(capacity, dtype=np.float32)
        self.next_obs_buffer = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.done_buffer = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buffer[self.ptr] = obs
        self.act_buffer[self.ptr] = act  # Store action as scalar integer
        self.rew_buffer[self.ptr] = rew
        self.next_obs_buffer[self.ptr] = next_obs
        self.done_buffer[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size, seq_len):
        if self.size < seq_len:
            raise ValueError(f"Not enough data. Buffer size: {self.size}, Required: {seq_len}")

        indices = np.random.randint(0, self.size - seq_len + 1, size=batch_size)

        obs_batch = np.array([self._get_seq(self.obs_buffer, idx, seq_len) for idx in indices])
        act_batch = np.array([self._get_seq(self.act_buffer, idx, seq_len) for idx in indices])
        rew_batch = np.array([self._get_seq(self.rew_buffer, idx, seq_len) for idx in indices])
        next_obs_batch = np.array([self._get_seq(self.next_obs_buffer, idx, seq_len) for idx in indices])
        done_batch = np.array([self._get_seq(self.done_buffer, idx, seq_len) for idx in indices])

        return (
            torch.as_tensor(obs_batch, dtype=torch.float32).to(device),
            torch.as_tensor(act_batch, dtype=torch.long).to(device),  # Actions as long tensors
            torch.as_tensor(rew_batch, dtype=torch.float32).to(device),
            torch.as_tensor(next_obs_batch, dtype=torch.float32).to(device),
            torch.as_tensor(done_batch, dtype=torch.float32).to(device)
        )

    def _get_seq(self, buffer, start_idx, seq_len):
        if start_idx + seq_len <= self.size:
            return buffer[start_idx:start_idx + seq_len]
        else:
            return np.concatenate((buffer[start_idx:], buffer[:seq_len - (self.size - start_idx)]), axis=0)

    def __len__(self):
        return self.size