import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import *


class ActionSpace:
    def __init__(self, action_dim):
        self.shape = (action_dim,)
        self.dtype = float
        self.low = -1.0
        self.high = 1.0


class TestNetworkShapes(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # The code expects observation in shape [B,T,C,H,W].
        self.B = 50
        self.T = 50
        self.obs_shape = (3, 64, 64)
        self.encoded_obs_size = 1024
        self.stoch_size = 30
        self.det_size = 200
        self.latent_size = self.stoch_size + self.det_size
        self.observations = torch.randn(self.B, self.T, *self.obs_shape)

        # Create dummy actions: assume continuous with dimension 6
        self.action_dim = 6
        self.action_space = ActionSpace(self.action_dim)
        self.actions = torch.randn(self.B, self.T, self.action_dim)

        self.world_model = WorldModel(self.action_dim).to(self.device)
        self.actor = Actor(self.action_space).to(self.device)
        self.critic = Critic().to(self.device)

        self.prev_action = torch.zeros(self.B, self.action_dim)
        self.prev_state = (
            torch.zeros(self.B, self.stoch_size).to(self.device),
            torch.zeros(self.B, self.det_size).to(self.device),
        )

    def test_observation_encoder(self):
        obs = torch.randn(self.B, self.T, *self.obs_shape).to(self.device)
        encoded = self.world_model.encoder(obs)
        expected = (self.B, self.T, self.encoded_obs_size)
        self.assertEqual(
            encoded.shape, expected, f"Expected shape {expected}, got {encoded.shape}"
        )

    def test_observation_decoder(self):
        latent = torch.randn(self.B, self.T, self.latent_size).to(self.device)
        # print("Latent shape:", latent.shape)
        decoded = self.world_model.decoder(latent)
        self.assertEqual(
            decoded.mean.shape,
            (self.B, self.T, *self.obs_shape),
            "Decoder mean shape mismatch",
        )
        # print("Variance shape:", decoded.variance.shape)
        self.assertEqual(
            decoded.variance.mean(), 1.0, "Decoder variance is not fixed at 1.0"
        )

    def test_rssm_output(self):
        # RSSM forward: Input: prev_state ((B,stoch),(B,det)), prev_action (B, A), obs: [B,T,...]
        encoded_obs = self.world_model.encoder(self.observations)
        (prior, posterior), state = self.world_model.rssm(
            self.prev_state, self.prev_action, encoded_obs[:, 0]
        )
        # state: (stoch, det) each [B,D]
        self.assertEqual(state[0].shape, (self.B, self.stoch_size))
        self.assertEqual(state[1].shape, (self.B, self.det_size))

    def test_prior(self):
        action = torch.randn(self.B, self.action_dim).to(self.device)
        dist, state = self.world_model.rssm.prior(self.prev_state, action)
        self.assertEqual(
            dist.mean.shape, (self.B, self.stoch_size), "Prior mean shape mismatch"
        )

    def test_posterior(self):
        obs = torch.randn(self.B, 1, self.encoded_obs_size).to(self.device)
        dist, (stoch, det) = self.world_model.rssm.posterior(self.prev_state, obs)
        self.assertEqual(stoch.shape[-1], self.stoch_size)
        self.assertEqual(det.shape[-1], self.det_size)
        self.assertEqual(
            dist.mean.shape, (self.B, self.stoch_size), "Posterior mean shape mismatch"
        )

    def test_world_model_observe(self):
        # world_model.observe: Input obs, actions -> (prior, posterior), features, decoded_dist, reward_dist, terminal_dist
        (_, _), features, decoded_dist, reward_dist, terminal_dist = (
            self.world_model.observe(self.observations, self.actions)
        )
        # features: [B, T,stoch+det]
        self.assertEqual(features.shape[0], self.B)
        self.assertEqual(features.shape[1], self.T)
        self.assertEqual(features.shape[2], self.latent_size)
        reward = reward_dist.rsample()
        terminal = terminal_dist.sample()
        decoded = decoded_dist.rsample()
        self.assertEqual(reward.shape, (self.B, self.T, 1))
        self.assertEqual(terminal.shape, (self.B, self.T, 1))
        self.assertEqual(decoded.shape, (self.B, self.T, *self.obs_shape))

    def test_actor_forward(self):
        # Actor: Input features: [B, stoch+det]
        feats = torch.randn(self.B, self.latent_size)
        dist = self.actor(feats)
        if self.actor.is_continuous:  # Normal distribution
            action = dist.rsample()
        else:  # Categorical distribution
            action = dist.sample()
        self.assertEqual(action.shape, (self.B, self.action_dim))

    def test_critic_forward(self):
        # Critic: Input features: [B, stoch+det]
        feats = torch.randn(self.B, self.latent_size)
        dist = self.critic(feats)
        value = dist.rsample()
        self.assertEqual(value.shape, (self.B, 1))


if __name__ == "__main__":
    unittest.main()
