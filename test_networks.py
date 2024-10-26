import unittest
import torch
from networks import ConvEncoder, ConvDecoder, WorldModel

class MockConfig:
    def __init__(self):
        self.hidden_dim = 512
        self.latent_dim = 32
        self.latent_categories = 32
        self.kl_balance_alpha = 0.8
        self.free_nats = 3.0
        self.actor_temperature = 1.0
        self.tau = 0.005

class TestNetworks(unittest.TestCase):
    def setUp(self):
        self.config = MockConfig()
        self.obs_shape = (4, 42, 42)
        self.act_dim = 6
        self.batch_size = 10
        self.seq_len = 5

    def test_conv_encoder(self):
        encoder = ConvEncoder(self.obs_shape, self.config)
        obs = torch.randn(self.batch_size, *self.obs_shape)
        encoded_obs = encoder(obs)
        self.assertEqual(encoded_obs.shape, (self.batch_size, self.config.hidden_dim))

    def test_conv_decoder(self):
        decoder = ConvDecoder(self.obs_shape, self.config)
        latent_input = torch.randn(self.batch_size, self.config.hidden_dim + 
                                   self.config.latent_dim * self.config.latent_categories)
        recon_obs = decoder(latent_input)
        self.assertEqual(recon_obs.shape, (self.batch_size, *self.obs_shape))

    def test_world_model_forward(self):
        world_model = WorldModel(self.obs_shape, self.act_dim, is_image=True, is_discrete=True, config=self.config)
        obs_seq = torch.randn(self.batch_size, self.seq_len, *self.obs_shape)
        act_seq = torch.randint(0, self.act_dim, (self.batch_size, self.seq_len))
        outputs = world_model(obs_seq, act_seq, tau=1.0)

        self.assertIn("recon_obs", outputs)
        self.assertIn("pred_reward", outputs)
        self.assertIn("pred_discount", outputs)
        self.assertIn("kl_loss", outputs)
        
        self.assertEqual(outputs["recon_obs"].shape, (self.batch_size, self.seq_len, *self.obs_shape))
        self.assertEqual(outputs["pred_reward"].shape, (self.batch_size, self.seq_len))
        self.assertEqual(outputs["pred_discount"].shape, (self.batch_size, self.seq_len))

if __name__ == "__main__":
    unittest.main()
