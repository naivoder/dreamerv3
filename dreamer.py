import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from torchvision import transforms
from collections import deque
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio

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
    return torch.sign(x) * torch.log(torch.abs(x) + 1 + 1e-6)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1 + 1e-6)


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



# Convolutional Encoder for Image Observations
class ConvEncoder(nn.Module):
    def __init__(self, obs_shape, config):
        super(ConvEncoder, self).__init__()
        c, h, w = obs_shape
        self.conv_net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=4, stride=2),  # Output: (32, h/2, w/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Output: (64, h/4, w/4)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),  # Output: (128, h/8, w/8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),  # Output: (256, h/16, w/16)
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
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),  # Output: (128, h/8, w/8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),  # Output: (64, h/4, w/4)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),  # Output: (32, h/2, w/2)
            nn.ReLU(),
            nn.ConvTranspose2d(32, c, kernel_size=4, stride=2),  # Output: (c, h, w)
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
        self.reward_decoder = nn.Linear(
            config.hidden_dim + config.latent_dim * config.latent_categories, 1
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
            rnn_input = torch.cat([posterior_sample_flat, act_seq_onehot[:, t]], dim=-1)  # Corrected dimensions

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

        # Decode observation and reward
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

        outputs = {
            "recon_obs": recon_obs,
            "pred_reward": pred_reward,
            "kl_loss": kl_loss,
            "rnn_h": h_seq,
            "posterior_sample": posterior_samples,
        }
        return outputs


    def compute_kl_loss(self, prior_logits, posterior_logits):
        prior_dist = torch.distributions.Categorical(logits=prior_logits)
        posterior_dist = torch.distributions.Categorical(logits=posterior_logits)

        kl_div_forward = torch.sum(posterior_dist.probs * (posterior_dist.logits - prior_dist.logits), dim=-1)
        kl_div_reverse = torch.sum(prior_dist.probs * (prior_dist.logits - posterior_dist.logits), dim=-1)

        kl_loss = self.kl_balance_alpha * kl_div_forward + (1 - self.kl_balance_alpha) * kl_div_reverse
        kl_loss = kl_loss.sum(dim=-1)  # Sum over latent dimensions
        kl_loss = torch.clamp(kl_loss - self.free_nats, min=0.0).mean() # Mean over batch and time steps
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


# DreamerV3 Agent
class DreamerV3:
    def __init__(self, obs_shape, act_dim, is_image, config):
        self.config = config
        self.obs_shape = obs_shape
        self.act_dim = act_dim
        self.is_image = is_image

        self.world_model = WorldModel(obs_shape, act_dim, is_image, config).to(device)
        self.actor = Actor(config.hidden_dim, act_dim, config).to(device)
        self.critic = Critic(config.hidden_dim, config).to(device)
        self.target_critic = Critic(config.hidden_dim, config).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.obs_normalizer = ObsNormalizer(obs_shape)

        self.world_optimizer = optim.AdamW(
            self.world_model.parameters(), lr=config.world_lr, weight_decay=config.weight_decay
        )
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(), lr=config.actor_lr, weight_decay=config.weight_decay
        )
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(), lr=config.critic_lr, weight_decay=config.weight_decay
        )

        self.replay_buffer = ReplayBuffer(config.replay_buffer_capacity, obs_shape, act_dim)

        # Hidden states
        self.h = None
        self.reset_hidden_states()

        # Temperature for Gumbel-Softmax
        self.temperature = config.init_temperature

        # Initialize networks
        self.world_model.apply(self.init_weights)
        self.actor.apply(self.init_weights)
        self.critic.apply(self.init_weights)
        self.target_critic.apply(self.init_weights)

        # Add observation normalizer if not using image observations
        if not is_image:
            self.obs_normalizer = ObsNormalizer(obs_shape)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            fan_in = m.weight.shape[1]
            fan_out = m.weight.shape[0]
            limit = np.sqrt(6 / (fan_in + fan_out))
            nn.init.uniform_(m.weight, -limit, limit)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRUCell):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def reset_hidden_states(self):
        self.h = torch.zeros(1, 1, self.config.hidden_dim, device=device)

    def update_world_model(self):
        batch = self.replay_buffer.sample_batch(
            self.config.batch_size, self.config.seq_len
        )
        if batch[0] is None:
            return  # Not enough data to train

        obs_seq, act_seq, rew_seq, _, _ = batch

        outputs = self.world_model(obs_seq, act_seq, tau=self.temperature)
        recon_obs = outputs["recon_obs"]
        pred_reward = outputs["pred_reward"]
        kl_loss = outputs["kl_loss"]


        # Reconstruction loss
        if self.is_image:
            recon_loss = F.mse_loss(recon_obs, obs_seq, reduction='none')
            recon_loss = recon_loss.mean(dim=[2, 3, 4])  # Mean over image dimensions
        else:
            recon_loss = F.mse_loss(recon_obs, obs_seq, reduction='none')
            recon_loss = recon_loss.mean(dim=2)  # Mean over observation dimensions

        recon_loss = recon_loss.mean()  # Mean over batch and sequence length

        # Reward prediction loss
        reward_loss = F.mse_loss(pred_reward, symlog(rew_seq), reduction='none')
        reward_loss = reward_loss.mean()

        # Total loss
        loss_world = recon_loss + reward_loss + self.config.kl_scale * kl_loss

        self.world_optimizer.zero_grad()
        loss_world.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.max_grad_norm)
        self.world_optimizer.step()

        return loss_world.item()


    def update_actor_and_critic(self):
        # Imagined rollouts
        batch = self.replay_buffer.sample_batch(self.config.batch_size, 1)
        if batch[0] is None:
            return  # Not enough data to train

        obs_seq, act_seq, _, _, _ = batch
        obs = obs_seq[:, 0]

        # Initialize hidden state with zeros
        imag_h = torch.zeros(1, self.config.batch_size, self.config.hidden_dim, device=device)

        # Encode observation
        obs_encoded = self.world_model.obs_encoder(obs)

        # Initialize imag_s with posterior sample from initial obs
        posterior_input = torch.cat([imag_h.squeeze(0), obs_encoded], dim=-1)
        posterior_logits = self.world_model.posterior_net(posterior_input)
        posterior_logits = posterior_logits.view(
            self.config.batch_size, self.config.latent_dim, self.config.latent_categories
        )
        imag_s = gumbel_softmax(posterior_logits, tau=self.temperature, hard=False).view(self.config.batch_size, -1)

        imag_states = []
        imag_rewards = []
        imag_action_log_probs = []

        for _ in range(self.config.imagination_horizon):
            imag_states.append(imag_h.squeeze(0))

            # Get action probabilities from actor
            action_probs = self.actor(imag_h.squeeze(0))
            action_dist = torch.distributions.Categorical(probs=action_probs)
            imag_action = action_dist.sample()
            imag_action_log_probs.append(action_dist.log_prob(imag_action))

            # Update imag_h and imag_s using the world model
            act_onehot = F.one_hot(imag_action, num_classes=self.act_dim).float()
            rnn_input = torch.cat([imag_s, act_onehot], dim=-1)
            rnn_input = rnn_input.unsqueeze(1)  # Shape: [batch_size, 1, input_size]

            # RNN step
            _, imag_h = self.world_model.rnn(rnn_input, imag_h)

            # Compute prior logits and sample imag_s
            prior_logits = self.world_model.prior_net(imag_h.squeeze(0))
            prior_logits = prior_logits.view(
                self.config.batch_size, self.config.latent_dim, self.config.latent_categories
            )
            imag_s = gumbel_softmax(prior_logits, tau=self.temperature, hard=False).view(self.config.batch_size, -1)

            # Predict reward
            decoder_input = torch.cat([imag_h.squeeze(0), imag_s], dim=-1)
            pred_reward = self.world_model.reward_decoder(decoder_input)
            pred_reward = symexp(pred_reward.squeeze(-1))
            imag_rewards.append(pred_reward)

        # Convert lists to tensors
        imag_states = torch.stack(imag_states)  
        imag_rewards = torch.stack(imag_rewards)  
        imag_action_log_probs = torch.stack(imag_action_log_probs)

        # Reshape imag_states for critic input
        imag_states_flat = imag_states.view(-1, self.config.hidden_dim)

        # Get value estimates for critic update (detach to prevent backprop through the world model)
        value_pred = self.critic(imag_states_flat.detach()).view(self.config.imagination_horizon, self.config.batch_size)

        # Calculate lambda-returns
        lambda_returns = torch.zeros_like(imag_rewards)
        last_value = value_pred[-1]
        for t in reversed(range(self.config.imagination_horizon)):
            if t == self.config.imagination_horizon - 1:
                bootstrap = last_value
            else:
                bootstrap = (1 - self.config.lambda_) * value_pred[t + 1] + self.config.lambda_ * lambda_returns[t + 1]
            lambda_returns[t] = imag_rewards[t] + self.config.gamma * bootstrap

        # Critic update
        value_target = lambda_returns.detach()
        critic_loss = F.mse_loss(value_pred, value_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        # Actor update
        # Recompute value estimates without detaching imag_states to allow gradients to flow
        value_pred_actor = self.critic(imag_states_flat).view(self.config.imagination_horizon, self.config.batch_size)
        advantage = (lambda_returns.detach() - value_pred_actor)
        actor_loss = -(advantage * imag_action_log_probs).mean()

        # Entropy regularization
        action_probs = self.actor(imag_states_flat)
        action_probs = action_probs.view(self.config.imagination_horizon, self.config.batch_size, -1)
        action_log_probs = torch.log(action_probs + 1e-8)
        entropy = -torch.sum(action_probs * action_log_probs, dim=-1).mean()
        actor_loss += -self.config.entropy_scale * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        # Update target critic
        self._soft_update(self.target_critic, self.critic)

        return actor_loss.item(), critic_loss.item()


    def _soft_update(self, target, source, tau=None):
        if tau is None:
            tau = self.config.tau
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def act(self, obs, reset=False):
        if reset:
            self.reset_hidden_states()

        if self.is_image:
            obs = torch.tensor(obs).float().to(device).unsqueeze(0) / 255.0
        else:
            obs = self.obs_normalizer.normalize(obs)
            obs = torch.tensor(obs).float().to(device).unsqueeze(0)

        with torch.no_grad():
            # Encode observation
            obs_encoded = self.world_model.obs_encoder(obs)

            # Compute posterior over latent variables
            h = self.h.squeeze(0)  # Shape: [1, hidden_dim]
            posterior_input = torch.cat([h, obs_encoded], dim=-1)
            posterior_logits = self.world_model.posterior_net(posterior_input)
            posterior_logits = posterior_logits.view(
                1, self.config.latent_dim, self.config.latent_categories
            )
            posterior_sample = gumbel_softmax(posterior_logits, tau=self.temperature, hard=False).view(1, -1)

            # Get action probabilities from actor
            action_probs = self.actor(h)
            action_dist = torch.distributions.Categorical(probs=action_probs)
            action = action_dist.sample().cpu().numpy()[0]

            # Update hidden state
            act_onehot = F.one_hot(
                torch.tensor([action], device=device), num_classes=self.act_dim
            ).float()
            rnn_input = torch.cat([posterior_sample, act_onehot], dim=-1)
            _, self.h = self.world_model.rnn(rnn_input.unsqueeze(1), self.h)

        return action


    def store_transition(self, obs, act, rew, next_obs, done):
        if not self.is_image:
            obs = self.obs_normalizer.normalize(obs)
            next_obs = self.obs_normalizer.normalize(next_obs)
            self.obs_normalizer.update(obs)

        self.replay_buffer.store(obs, act, rew, next_obs, done)

    def train(self, num_updates):
        world_losses = []
        actor_losses = []
        critic_losses = []

        for _ in range(num_updates):
            world_loss = self.update_world_model()
            actor_loss, critic_loss = self.update_actor_and_critic()
            
            world_losses.append(world_loss)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            # Anneal temperature
            self.temperature = max(self.temperature * self.config.temperature_decay, self.config.min_temperature)

        return {
            'world_loss': np.mean(world_losses),
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses)
        }



def train_dreamer(config):
    env = gym.make(config.env)
    # Determine if the observation space is image-based or vector-based
    obs_space = env.observation_space
    is_image = isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3

    if is_image:
        # Preprocess observations to shape (3, 64, 64)
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )
        obs_shape = (3, 64, 64)
    else:
        obs_shape = obs_space.shape
        transform = None

    act_dim = env.action_space.n
    agent = DreamerV3(obs_shape, act_dim, is_image, config)
    total_rewards = []
    world_losses = [] 
    actor_losses = []  
    critic_losses = []

    frame_idx = 0  # For temperature annealing
    avg_reward_window = 100  # Running average over the last 100 episodes
    best_avg_reward = float('-inf')
    best_weights = None

    with tqdm(total=config.episodes, desc="Training Progress", unit="episode") as pbar:
        for episode in range(config.episodes):
            obs, _ = env.reset()
            if is_image:
                obs = transform(obs).numpy()
            else:
                obs = obs.astype(np.float32)
            done = False
            episode_reward = 0
            agent.act(obs, reset=True)  # Reset hidden states at episode start

            while not done:
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if is_image:
                    next_obs_processed = transform(next_obs).numpy()
                    agent.store_transition(obs, action, reward, next_obs_processed, done)
                    obs = next_obs_processed
                else:
                    next_obs = next_obs.astype(np.float32)
                    agent.store_transition(obs, action, reward, next_obs, done)
                    obs = next_obs

                episode_reward += reward
                frame_idx += 1

                if len(agent.replay_buffer) > config.min_buffer_size:
                    if frame_idx % config.train_horizon == 0:
                        losses = agent.train(num_updates=config.num_updates)
                        world_losses.append(losses['world_loss'])
                        actor_losses.append(losses['actor_loss'])
                        critic_losses.append(losses['critic_loss'])

                if done:
                    total_rewards.append(episode_reward)
                    # Update the progress bar with running average reward
                    if len(total_rewards) >= avg_reward_window:
                        running_avg_reward = sum(total_rewards[-avg_reward_window:]) / avg_reward_window
                    else:
                        running_avg_reward = sum(total_rewards) / len(total_rewards)
                    
                    pbar.set_postfix({"Running Avg. Reward": f"{running_avg_reward:.2f}"})
                    pbar.update(1)  # Update the progress bar for one episode completion
                    break

            if len(total_rewards) >= avg_reward_window:
                running_avg_reward = sum(total_rewards[-avg_reward_window:]) / avg_reward_window
                if running_avg_reward > best_avg_reward:
                    best_avg_reward = running_avg_reward
                    best_weights = {
                        'world_model': agent.world_model.state_dict(),
                        'actor': agent.actor.state_dict(),
                        'critic': agent.critic.state_dict()
                    }

    # Plot losses and rewards
    plot_results(total_rewards, world_losses, actor_losses, critic_losses)

    # Create animation
    create_animation(env, agent, best_weights, config)

    return total_rewards

def plot_results(rewards, world_losses, actor_losses, critic_losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot rewards
    ax1.plot(rewards, label='Episode Reward')
    ax1.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), label='Running Average (100 episodes)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Rewards over Episodes')
    ax1.legend()

    # Plot losses
    ax2.plot(world_losses, label='World Model Loss')
    ax2.plot(actor_losses, label='Actor Loss')
    ax2.plot(critic_losses, label='Critic Loss')
    ax2.set_xlabel('Update Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Losses over Update Steps')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('dreamer_results.png')
    plt.close()

def create_animation(env, agent, best_weights, config):
    # Load best weights
    agent.world_model.load_state_dict(best_weights['world_model'])
    agent.actor.load_state_dict(best_weights['actor'])
    agent.critic.load_state_dict(best_weights['critic'])

    obs, _ = env.reset()
    frames = []

    for _ in range(1000):  # Adjust the number of steps as needed
        frames.append(env.render())
        action = agent.act(obs)
        obs, _, done, _, _ = env.step(action)
        if done:
            break

    env.close()

    # Save animation as GIF
    imageio.mimsave('dreamer_animation.gif', np.array(frames), fps=30)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--train_horizon", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--latent_categories", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=512)  
    parser.add_argument("--actor_lr", type=float, default=3e-5) 
    parser.add_argument("--critic_lr", type=float, default=3e-5) 
    parser.add_argument("--world_lr", type=float, default=1e-4)  
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--imagination_horizon", type=int, default=15)  
    parser.add_argument("--free_nats", type=float, default=3.0)  
    parser.add_argument("--batch_size", type=int, default=250)  
    parser.add_argument("--seq_len", type=int, default=50) 
    parser.add_argument("--replay_buffer_capacity", type=int, default=1000000)  
    parser.add_argument("--entropy_scale", type=float, default=1e-3)  
    parser.add_argument("--kl_balance_alpha", type=float, default=0.8)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)  
    parser.add_argument("--weight_decay", type=float, default=0.0)  
    parser.add_argument("--num_updates", type=int, default=1)  
    parser.add_argument("--min_buffer_size", type=int, default=5000) 
    parser.add_argument("--init_temperature", type=float, default=0.1)  
    parser.add_argument("--temperature_decay", type=float, default=1.0)  
    parser.add_argument("--min_temperature", type=float, default=0.1)
    parser.add_argument("--actor_temperature", type=float, default=0.1) 
    parser.add_argument("--tau", type=float, default=0.02) 
    parser.add_argument("--kl_scale", type=float, default=1.0)
    args = parser.parse_args()

    train_dreamer(args)
