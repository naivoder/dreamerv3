import unittest
import gymnasium as gym
import numpy as np
from memory import ReplayBuffer
from types import SimpleNamespace
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    @staticmethod
    def make(d):
        """
        Recursively convert a dictionary to a SimpleNamespace.
        """
        if isinstance(d, dict):
            return SimpleNamespace(**{k: Config.make(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [Config.make(v) for v in d]
        else:
            return d


def interact(env, episodes, episode_length, buffer):
    env = gym.wrappers.TimeLimit(env, max_episode_steps=episode_length)
    for _ in range(episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_observation, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            buffer.store(
                observation,
                action.astype(np.float32),
                np.array(reward, np.float32),
                np.array(done, np.bool),
            )
            observation = next_observation


class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Pendulum-v1")
        self.episodes = 3
        self.max_length = 10
        self.config = Config.make(
            {"capacity": 5, "batch_size": 2, "sequence_length": 4}
        )

    def test_store(self):
        buffer = ReplayBuffer(self.config, DEVICE)
        interact(self.env, self.episodes, self.max_length, buffer)
        self.assertEqual(len(buffer.episodes), self.episodes)

    def test_sample(self):
        buffer = ReplayBuffer(self.config, DEVICE)
        interact(self.env, self.episodes, self.max_length, buffer)
        sample = next(iter(buffer.sample(1)))
        self.assertEqual(sample["observation"].shape[0], 2)
        self.assertEqual(sample["observation"].shape[1], 4)


if __name__ == "__main__":
    unittest.main()
