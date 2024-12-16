import torch
import numpy as np
from collections import deque
import utils

class ReplayBuffer:
    def __init__(self, config, device):
        self.capacity = config.capacity
        self.batch_size = config.batch_size
        self.seq_length = config.sequence_length
        self.device = device

        self.episodes = deque(maxlen=self.capacity)

        # Temporary storage for the current episode
        self._ep = {"observation": [], "action": [], "reward": [], "done": []}

    def store(self, obs, act, rew, next_obs, done):
        self._ep["observation"].append(obs)
        self._ep["action"].append(act)
        self._ep["reward"].append(rew)
        self._ep["done"].append(done)
        
        if done:
            # Add bootstrap observation
            self._ep["observation"].append(next_obs)

            # Convert lists to NumPy arrays
            ep = {k: np.array(v) for k, v in self._ep.items()}
            ep["observation"] = utils.quantize(ep["observation"])  

            self.episodes.append(ep) 
            self._ep = {"observation": [], "action": [], "reward": [], "done": []}  # Reset for new episode

    def sample(self, n_batches):
        for _ in range(n_batches):
            batch = {k: [] for k in ["observation", "action", "reward", "done"]}
            for _ in range(self.batch_size):
                ep = np.random.choice(self.episodes)
                t = np.random.randint(len(ep["observation"]) - self.seq_length)
                for k in batch.keys():
                    batch[k].append(ep[k][t : t + self.seq_length])

            for k in batch.keys():
                batch[k] = torch.tensor(np.array(batch[k]), device=self.device, requires_grad=False)

            batch["observation"] = utils.preprocess(batch["observation"])
            yield batch

    def __len__(self):
        return len(self.episodes)
