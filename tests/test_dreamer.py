import sys

sys.path.append("/Users/naivoder/Code/dreamerv3/")
import unittest
import torch
from unittest.mock import MagicMock
from dreamer import Dreamer
from networks import WorldModel, Actor, Critic
from memory import ReplayBuffer
from types import SimpleNamespace
import yaml


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


class TestDreamer(unittest.TestCase):
    def setUp(self):
        self.obs_shape = (3, 64, 64)  # Example observation shape (C, H, W)
        self.act_space = MagicMock()
        self.act_space.shape = (4,)
        self.act_space.low = -1
        self.act_space.high = 1
        self.act_space.sample = lambda: torch.zeros(self.act_space.shape)

        self.config = Config.load_from_yaml("config.yaml")

        self.dreamer = Dreamer(self.obs_shape, self.act_space, self.config)

    def test_initialization(self):
        self.assertIsInstance(self.dreamer.world_model, WorldModel)
        self.assertIsInstance(self.dreamer.actor, Actor)
        self.assertIsInstance(self.dreamer.critic, Critic)
        self.assertIsInstance(self.dreamer.memory, ReplayBuffer)

    def test_init_state(self):
        state = self.dreamer.init_state()
        self.assertEqual(state[0][0].shape, (1, self.config.rssm["stochastic_size"]))
        self.assertEqual(state[0][1].shape, (1, self.config.rssm["deterministic_size"]))
        self.assertEqual(state[1].shape, (self.act_space.shape))

    def test_random_action(self):
        action = self.dreamer.random_action()
        self.assertEqual(action.shape, self.act_space.shape)

    def test_act(self):
        prev_state = self.dreamer.init_state()
        prev_action = torch.zeros(self.act_space.shape)
        obs = torch.zeros(self.obs_shape)

        state, action = self.dreamer.act(prev_state, prev_action, obs, training=True)
        self.assertEqual(state[0].shape[-1], self.config.rssm["stochastic_size"])
        self.assertEqual(state[1].shape[-1], self.config.rssm["deterministic_size"])
        self.assertEqual(action.shape, prev_action.shape)

    def test_learn(self):
        self.dreamer.memory.sample = MagicMock(
            return_value=[
                {
                    "observation": torch.zeros((50, 5) + self.obs_shape),
                    "action": torch.zeros((50, 5) + self.act_space.shape),
                    "reward": torch.zeros((50, 5, 1)),
                    "done": torch.zeros((50, 5, 1)),
                }
            ]
        )
        self.dreamer.update = MagicMock(return_value={"loss": torch.tensor(0.1)})

        self.dreamer.learn()
        self.assertGreater(self.dreamer.step, 0)

    def test_save_and_load_checkpoint(self):
        # Save weights
        self.dreamer.save_checkpoint()
        # Modify weights
        for param in self.dreamer.world_model.parameters():
            param.data.add_(1.0)
        for param in self.dreamer.actor.parameters():
            param.data.add_(1.0)
        for param in self.dreamer.critic.parameters():
            param.data.add_(1.0)
        # Load weights back
        self.dreamer.load_checkpoint()

        # Check that parameters match initial values
        for param in self.dreamer.world_model.parameters():
            self.assertTrue((param.data == 1.0).sum() == 0)
        for param in self.dreamer.actor.parameters():
            self.assertTrue((param.data == 1.0).sum() == 0)
        for param in self.dreamer.critic.parameters():
            self.assertTrue((param.data == 1.0).sum() == 0)


if __name__ == "__main__":
    unittest.main()
