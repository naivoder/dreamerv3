import torch.nn as nn
import torch.nn.functional as F
import torch

from utils import gumbel_softmax

class ConvEncoder(nn.Module):
    def __init__(self, obs_shape, config):
        super(ConvEncoder, self).__init__()
        c, h, w = obs_shape
        self.conv_net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, h/2, w/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, h/4, w/4)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, h/8, w/8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: (256, h/16, w/16)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * (h // 16) * (w // 16), config.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        obs = obs / 255.0  # Normalize pixel values
        return self.conv_net(obs)


# Convolutional Decoder for Reconstructing Image Observations
class ConvDecoder(nn.Module):
    def __init__(self, obs_shape, config):
        super(ConvDecoder, self).__init__()
        self.obs_shape = obs_shape
        c, h, w = obs_shape
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim + config.latent_dim * config.latent_categories,
                      256 * (h // 16) * (w // 16)),
            nn.ReLU(),
        )
        self.deconv_net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, h/8, w/8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, h/4, w/4)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, h/2, w/2)
            nn.ReLU(),
            nn.ConvTranspose2d(32, c, kernel_size=4, stride=2, padding=1),  # Output: (c, h, w)
            nn.Sigmoid(),  # Output pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.fc(x)
        batch_size = x.size(0)
        c, h, w = 256, self.obs_shape[1] // 16, self.obs_shape[2] // 16
        x = x.view(batch_size, c, h, w)
        return self.deconv_net(x)


# MLP Encoder for Vector Observations
class MLPEncoder(nn.Module):
    def __init__(self, obs_dim, config):
        super(MLPEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs)


# MLP Decoder for Reconstructing Vector Observations
class MLPDecoder(nn.Module):
    def __init__(self, obs_dim, config):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim + config.latent_dim * config.latent_categories, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, obs_dim),
        )

    def forward(self, x):
        return self.net(x)


# World Model with Discrete Latent Representations and KL Balancing
class WorldModel(nn.Module):
    def __init__(self, obs_shape, act_dim, is_image, config):
        super(WorldModel, self).__init__()
        self.is_image = is_image
        self.act_dim = act_dim
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.latent_categories = config.latent_categories
        self.kl_balance_alpha = config.kl_balance_alpha
        self.free_nats = config.free_nats

        if self.is_image:
            self.obs_shape = obs_shape
            self.obs_encoder = ConvEncoder(obs_shape, config)
            self.obs_decoder = ConvDecoder(obs_shape, config)
            self.obs_dim = None
        else:
            self.obs_dim = obs_shape[0]
            self.obs_encoder = MLPEncoder(self.obs_dim, config)
            self.obs_decoder = MLPDecoder(self.obs_dim, config)
            self.obs_shape = (self.obs_dim,)

        self.rnn = nn.GRU(
            config.latent_dim * config.latent_categories + act_dim, config.hidden_dim, batch_first=True
        )
        self.prior_net = nn.Linear(config.hidden_dim, config.latent_dim * config.latent_categories)
        self.posterior_net = nn.Linear(
            config.hidden_dim + config.hidden_dim, config.latent_dim * config.latent_categories
        )
        self.reward_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim + config.latent_dim * config.latent_categories, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.discount_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim + config.latent_dim * config.latent_categories, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),  # Output between 0 and 1
        )

    def forward(self, obs_seq, act_seq, tau):
        batch_size, seq_len = obs_seq.size(0), obs_seq.size(1)

        # Encode observations
        if self.is_image:
            obs_seq = obs_seq.view(batch_size * seq_len, *self.obs_shape)
        else:
            obs_seq = obs_seq.view(batch_size * seq_len, -1)
        obs_encoded = self.obs_encoder(obs_seq)
        obs_encoded = obs_encoded.view(batch_size, seq_len, -1)

        # One-hot encode actions
        act_seq_onehot = F.one_hot(act_seq, num_classes=self.act_dim).float()  # Shape: [batch_size, seq_len, act_dim]

        # Initialize hidden state
        h = torch.zeros(1, batch_size, self.hidden_dim, device=obs_seq.device)

        # RNN forward pass
        posterior_samples = []
        prior_logits_list = []
        posterior_logits_list = []
        h_list = []

        for t in range(seq_len):
            # Compute posterior
            posterior_input = torch.cat([h.transpose(0, 1).squeeze(1), obs_encoded[:, t]], dim=-1)
            posterior_logits = self.posterior_net(posterior_input)
            posterior_logits = posterior_logits.view(batch_size, self.latent_dim, self.latent_categories)
            posterior_sample = gumbel_softmax(posterior_logits, tau=tau, hard=False)
            posterior_sample_flat = posterior_sample.view(batch_size, -1)

            posterior_logits_list.append(posterior_logits)
            posterior_samples.append(posterior_sample_flat)

            # Prepare RNN input
            rnn_input = torch.cat([posterior_sample_flat.detach(), act_seq_onehot[:, t]], dim=-1)  # Stop gradient

            # RNN step
            _, h = self.rnn(rnn_input.unsqueeze(1), h)

            # Store hidden state
            h_list.append(h.transpose(0, 1).squeeze(1))

            # Compute prior
            prior_logits = self.prior_net(h.transpose(0, 1).squeeze(1))
            prior_logits = prior_logits.view(batch_size, self.latent_dim, self.latent_categories)
            prior_logits_list.append(prior_logits)

        # Stack tensors
        posterior_samples = torch.stack(posterior_samples, dim=1)
        prior_logits = torch.stack(prior_logits_list, dim=1)
        posterior_logits = torch.stack(posterior_logits_list, dim=1)
        h_seq = torch.stack(h_list, dim=1)  # h_seq shape: [batch_size, seq_len, hidden_dim]

        # Compute KL divergence
        kl_loss = self.compute_kl_loss(prior_logits, posterior_logits)

        # Decode observation, reward, and discount
        decoder_input = torch.cat([h_seq, posterior_samples], dim=-1)  # [batch_size, seq_len, hidden_dim + latent_dim * latent_categories]
        decoder_input_flat = decoder_input.view(batch_size * seq_len, -1)

        if self.is_image:
            recon_obs = self.obs_decoder(decoder_input_flat)
            recon_obs = recon_obs.view(batch_size, seq_len, *self.obs_shape)
        else:
            recon_obs = self.obs_decoder(decoder_input_flat)
            recon_obs = recon_obs.view(batch_size, seq_len, *self.obs_shape)

        pred_reward = self.reward_decoder(decoder_input_flat)
        pred_reward = pred_reward.view(batch_size, seq_len)

        pred_discount = self.discount_decoder(decoder_input_flat)
        pred_discount = pred_discount.view(batch_size, seq_len)

        outputs = {
            "recon_obs": recon_obs,
            "pred_reward": pred_reward,
            "pred_discount": pred_discount,
            "kl_loss": kl_loss,
            "rnn_h": h_seq,
            "posterior_sample": posterior_samples,
        }
        return outputs

    def compute_kl_loss(self, prior_logits, posterior_logits):
        prior_dist = torch.distributions.Categorical(logits=prior_logits)
        posterior_dist = torch.distributions.Categorical(logits=posterior_logits)

        # Stop gradient to prior in rep_loss
        prior_logits_sg = prior_logits.detach()
        prior_dist_sg = torch.distributions.Categorical(logits=prior_logits_sg)
        rep_loss = torch.distributions.kl_divergence(posterior_dist, prior_dist_sg)

        # Stop gradient to posterior in dyn_loss
        posterior_logits_sg = posterior_logits.detach()
        posterior_dist_sg = torch.distributions.Categorical(logits=posterior_logits_sg)
        dyn_loss = torch.distributions.kl_divergence(posterior_dist_sg, prior_dist)

        kl_loss = self.kl_balance_alpha * dyn_loss + (1 - self.kl_balance_alpha) * rep_loss
        kl_loss = kl_loss.sum(dim=-1)  # Sum over latent dimensions
        kl_loss = torch.clamp(kl_loss - self.free_nats, min=0.0).mean()  # Mean over batch and time steps
        return kl_loss


# Actor Network for Discrete Actions with Temperature Scaling
class Actor(nn.Module):
    def __init__(self, state_dim, act_dim, config):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, act_dim),
        )
        self.temperature = config.actor_temperature

    def forward(self, x):
        logits = self.net(x)
        action_probs = torch.softmax(logits / self.temperature, dim=-1)
        return action_probs


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, config):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)