import torch
import numpy as np
import utils


class ReplayBuffer:
    def __init__(self, capacity, batch_size, seq_length, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.device = device

        self.episodes = []

        self._ep = {"observation": [], "action": [], "reward": [], "done": []}

    def store(self, obs, act, rew, done):
        self._ep["observation"].append(obs)
        self._ep["action"].append(act)
        self._ep["reward"].append(rew)
        self._ep["done"].append(done)
        if done:
            ep = {k: np.array(v) for k, v in self._ep.items()}
            ep["observation"] = utils.quantize(ep["observation"])

            self.episodes.append(ep)
            if len(self.episodes) > self.capacity:
                self.episodes.pop(0)

    def sample(self, n_batches):
        for _ in range(n_batches):
            batch = {"observation": [], "action": [], "reward": [], "done": []}
            for _ in range(self.batch_size):
                ep = np.random.choice(self.episodes)
                t = np.random.randint(len(ep["observation"]) - self.seq_length)
                for k in batch.keys():
                    batch[k].append(ep[k][t : t + self.seq_length])

            for k in batch.keys():
                batch[k] = torch.tensor(batch[k], device=self.device)

            batch["observation"] = utils.preprocess(batch["observation"])
            yield batch

    def __len__(self):
        return len(self.episodes)
