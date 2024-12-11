import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import *


class TestNetworkShapes(unittest.TestCase):
    def setUp(self):
        # Minimal config objects with necessary attributes for testing
        class EncoderConfig:
            def __init__(self):
                self.depth = 32
                self.kernels = [4, 4, 4, 4]
                self.output_dim = 1024

        class DecoderConfig:
            def __init__(self):
                self.depth = 32
                self.kernels = [5, 5, 6, 6]

        class RSSMConfig:
            def __init__(self):
                self.stochastic_size = 30
                self.deterministic_size = 200
                self.hidden_size = 200

        class ModelOpt:
            def __init__(self):
                self.lr = 1e-3
                self.eps = 1e-5
                self.clip = 100.0

        class ActorConfig:
            def __init__(self, action_space):
                self.min_stddev = 0.1
                self.lr = 1e-3
                self.eps = 1e-5

                is_continuous = (
                    action_space.dtype == float and len(action_space.shape) == 1
                )
                if is_continuous:
                    n_actions = action_space.shape[0] * 2
                else:
                    n_actions = action_space.n
                self.output_sizes = [400, 400, 400, n_actions]

        class CriticConfig:
            def __init__(self):
                self.min_stddev = 0.1
                self.lr = 1e-3
                self.eps = 1e-5
                self.output_sizes = [400, 400, 400, 1]

        class RewardDecoderConfig:
            def __init__(self):
                self.output_sizes = [230, 400, 400, 400, 1]

        class TerminalDecoderConfig:
            def __init__(self):
                self.output_sizes = [230, 400, 400, 400, 1]

        class CombinedConfig:
            def __init__(self, action_space):
                self.encoder = EncoderConfig()
                self.decoder = DecoderConfig()
                self.rssm = RSSMConfig()
                self.reward = RewardDecoderConfig()
                self.terminal = TerminalDecoderConfig()
                self.model_opt = ModelOpt()
                self.actor = ActorConfig(action_space)
                self.critic = CriticConfig()
                self.imag_horizon = 15

        # Create a dummy observation: B=2, T=1, C=4, H=64, W=64
        # The code expects observation in shape [B,T,C,H,W].
        self.B = 50
        self.T = 50
        self.C = 4
        self.H = 64
        self.W = 64
        self.observations = torch.randn(self.B, self.T, self.C, self.H, self.W)

        # Create dummy actions: assume continuous with dimension 6
        self.action_dim = 6
        self.actions = torch.randn(self.B, self.T, self.action_dim)

        # Define a dummy action space for Actor
        class ActionSpace:
            def __init__(self, action_dim):
                self.shape = (action_dim,)
                self.dtype = float
                self.low = -1.0
                self.high = 1.0

        action_space = ActionSpace(self.action_dim)

        self.config = CombinedConfig(action_space)

        self.encoder = ObservationEncoder((self.C, self.H, self.W), self.config.encoder)
        self.decoder = ObservationDecoder((self.C, self.H, self.W), self.config.decoder)
        self.rssm = RSSM(self.action_dim, self.config)
        self.world_model = WorldModel(
            (self.C, self.H, self.W), self.action_dim, self.config
        )
        self.actor = Actor(action_space, self.config)
        self.critic = Critic(self.config)

    def test_observation_encoder_output(self):
        # Encoder: Input [B,T,C,H,W] -> Output [B,T,-1]
        encoded = self.encoder(self.observations)
        B, T = self.B, self.T
        # Check that encoded features are indeed (B,T, some dimension)
        self.assertEqual(encoded.shape[0], B)
        self.assertEqual(encoded.shape[1], T)
        self.assertTrue(encoded.shape[2] > 0)

    def test_observation_decoder_output(self):
        # Decoder: Input [B,T,D] -> Output Distribution over [B,T,C,H,W]
        # Create a dummy feature vector
        features = torch.randn(self.B, self.T, 230)
        dist = self.decoder(features)
        loc = dist.base_dist.loc
        self.assertEqual(loc.shape, (self.B, self.T, self.C, self.H, self.W))

    def test_rssm_output(self):
        # RSSM forward: Input: prev_state ((B,stoch),(B,det)), prev_action (B, A), obs: [B,T,...]
        # prev_state
        stoch_size = self.config.rssm.stochastic_size
        det_size = self.config.rssm.deterministic_size
        prev_state = (torch.zeros(self.B, stoch_size), torch.zeros(self.B, det_size))
        prev_action = torch.zeros(self.B, self.action_dim)
        encoded_obs = self.encoder(self.observations)
        (prior, posterior), state = self.rssm(
            prev_state, prev_action, encoded_obs[:, 0]
        )
        # state: (stoch, det) each [B,D]
        self.assertEqual(state[0].shape, (self.B, stoch_size))
        self.assertEqual(state[1].shape, (self.B, det_size))

    def test_world_model_observe(self):
        # world_model.observe: Input obs, actions -> (prior, posterior), features, decoded_dist, reward_dist, terminal_dist
        obs = self.observations
        acts = self.actions
        (prior, posterior), features, decoded_dist, reward_dist, terminal_dist = (
            self.world_model.observe(obs, acts)
        )
        B, T = self.B, self.T
        # features: [B,T, stoch+det]
        self.assertEqual(features.shape[0], B)
        self.assertEqual(features.shape[1], T)
        self.assertEqual(
            features.shape[2],
            self.config.rssm.stochastic_size + self.config.rssm.deterministic_size,
        )
        # decoded_dist loc: [B,T,C,H,W]
        self.assertEqual(
            decoded_dist.base_dist.loc.shape, (B, T, self.C, self.H, self.W)
        )

    def test_actor_forward(self):
        # Actor: Input features: [B, stoch+det]
        features_dim = (
            self.config.rssm.stochastic_size + self.config.rssm.deterministic_size
        )
        feats = torch.randn(self.B, features_dim)
        dist = self.actor(feats)
        # For continuous: dist is Independent Normal Tanh transformed, check shape
        sample = dist.rsample()
        self.assertEqual(sample.shape, (self.B, self.action_dim))

    def test_critic_forward(self):
        # Critic: Input features: [B, stoch+det]
        features_dim = (
            self.config.rssm.stochastic_size + self.config.rssm.deterministic_size
        )
        feats = torch.randn(self.B, features_dim)
        value = self.critic(feats)
        self.assertEqual(value.shape, (self.B, 1))


if __name__ == "__main__":
    unittest.main()
