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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# Simplified Replay Buffer for sequential sampling
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(self, transition):
        self.buffer.append(transition)

    def sample_batch(self, batch_size, seq_len):
        if len(self.buffer) < seq_len:
            return [None] * 5  # Not enough data
        indices = np.random.randint(0, len(self.buffer) - seq_len + 1, size=batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = [], [], [], [], []
        for idx in indices:
            seq = list(itertools.islice(self.buffer, idx, idx + seq_len))
            obs_seq, act_seq, rew_seq, next_obs_seq, done_seq = zip(*seq)
            obs_batch.append(np.array(obs_seq))
            act_batch.append(np.array(act_seq))
            rew_batch.append(np.array(rew_seq))
            next_obs_batch.append(np.array(next_obs_seq))
            done_batch.append(np.array(done_seq))
        return (
            torch.tensor(np.array(obs_batch), dtype=torch.float32).to(device),
            torch.tensor(np.array(act_batch), dtype=torch.long).to(device),
            torch.tensor(np.array(rew_batch), dtype=torch.float32).to(device),
            torch.tensor(np.array(next_obs_batch), dtype=torch.float32).to(device),
            torch.tensor(np.array(done_batch), dtype=torch.float32).to(device),
        )

    def __len__(self):
        return len(self.buffer)


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

        if self.is_image:
            obs_seq = obs_seq.view(
                batch_size * seq_len, *self.obs_shape
            )  # [batch_size * seq_len, c, h, w]
            obs_encoded = self.obs_encoder(obs_seq)
        else:
            obs_seq = obs_seq.view(
                batch_size * seq_len, -1
            )  # [batch_size * seq_len, obs_dim]
            obs_encoded = self.obs_encoder(obs_seq)

        obs_encoded = obs_encoded.view(batch_size, seq_len, -1)

        act_seq_onehot = torch.nn.functional.one_hot(
            act_seq, num_classes=self.act_dim
        ).float()

        # Initialize hidden state
        h = torch.zeros(1, batch_size, self.hidden_dim, device=obs_seq.device)

        # Initialize posterior sample
        posterior_input = torch.cat([h.permute(1, 0, 2), obs_encoded[:, 0:1]], dim=-1)
        posterior_logits = self.posterior_net(posterior_input)
        posterior_logits = posterior_logits.view(
            batch_size, self.latent_dim, self.latent_categories
        )
        posterior_sample = gumbel_softmax(posterior_logits, tau=tau, hard=False).view(
            batch_size, 1, -1
        )

        # Prepare inputs for RNN
        rnn_inputs = []
        posterior_samples = [posterior_sample]
        rnn_hidden_states = []

        for t in range(seq_len - 1):
            rnn_input = torch.cat(
                [posterior_samples[-1], act_seq_onehot[:, t : t + 1]], dim=-1
            )
            rnn_inputs.append(rnn_input)
            _, h = self.rnn(rnn_input, h)
            rnn_hidden_states.append(h.permute(1, 0, 2))
            posterior_input = torch.cat(
                [h.permute(1, 0, 2), obs_encoded[:, t + 1 : t + 2]], dim=-1,
            )
            posterior_logits = self.posterior_net(posterior_input)
            posterior_logits = posterior_logits.view(
                batch_size, self.latent_dim, self.latent_categories
            )
            posterior_sample = gumbel_softmax(posterior_logits, tau=tau, hard=False).view(
                batch_size, 1, -1
            )
            posterior_samples.append(posterior_sample)

        rnn_inputs = torch.cat(
            rnn_inputs, dim=1
        )  # [batch_size, seq_len - 1, input_size]
        rnn_hidden_states = torch.cat(
            rnn_hidden_states, dim=1
        )  # [batch_size, seq_len - 1, hidden_dim]
        posterior_samples = torch.cat(posterior_samples[:-1], dim=1)

        # Compute prior logits
        prior_logits = self.prior_net(rnn_hidden_states)
        prior_logits = prior_logits.view(
            batch_size, seq_len - 1, self.latent_dim, self.latent_categories
        )

        # Compute KL divergence with KL balancing
        posterior_logits_list = []
        for t in range(seq_len - 1):
            posterior_input = torch.cat(
                [rnn_hidden_states[:, t : t + 1], obs_encoded[:, t + 1 : t + 2]],
                dim=-1,
            )
            logits = self.posterior_net(posterior_input)
            posterior_logits_list.append(logits)
        posterior_logits = torch.stack(posterior_logits_list, dim=1)
        posterior_logits = posterior_logits.view(
            batch_size, seq_len - 1, self.latent_dim, self.latent_categories
        )

        posterior_sample = posterior_samples.view(batch_size, seq_len - 1, -1)

        kl_loss = self.compute_kl_loss(prior_logits, posterior_logits)

        # Decode observation and reward
        decoder_input = torch.cat([rnn_hidden_states, posterior_sample], dim=-1)
        if self.is_image:
            recon_obs = self.obs_decoder(
                decoder_input.view(-1, decoder_input.size(-1))
            )
            recon_obs = recon_obs.view(batch_size, seq_len - 1, *self.obs_shape)
        else:
            recon_obs = self.obs_decoder(
                decoder_input.view(-1, decoder_input.size(-1))
            )
            recon_obs = recon_obs.view(batch_size, seq_len - 1, *self.obs_shape)
        pred_reward = self.reward_decoder(decoder_input)

        outputs = {
            "recon_obs": recon_obs,
            "pred_reward": pred_reward.squeeze(-1),
            "kl_loss": kl_loss,
            "rnn_h": rnn_hidden_states,
            "posterior_sample": posterior_sample,
        }
        return outputs

    def compute_kl_loss(self, prior_logits, posterior_logits):
        prior_dist = torch.distributions.Categorical(logits=prior_logits)
        posterior_dist = torch.distributions.Categorical(logits=posterior_logits)

        # Compute KL divergence per latent dimension and time step
        kl_div_forward = torch.distributions.kl_divergence(posterior_dist, prior_dist)
        # kl_div_forward shape: [batch_size, seq_len - 1, latent_dim]

        # Sum over latent dimensions (latent_dim)
        kl_div_forward = kl_div_forward.sum(dim=2)  # Shape: [batch_size, seq_len - 1]

        kl_div_reverse = torch.distributions.kl_divergence(prior_dist, posterior_dist)
        kl_div_reverse = kl_div_reverse.sum(dim=2)  # Shape: [batch_size, seq_len - 1]

        kl_loss = (
            self.kl_balance_alpha * kl_div_forward
            + (1 - self.kl_balance_alpha) * kl_div_reverse
        )
        kl_loss = torch.clamp(kl_loss - self.free_nats, min=0.0).mean(dim=1)  # Mean over time steps
        return kl_loss  # Shape: [batch_size]




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

        self.world_optimizer = optim.Adam(
            self.world_model.parameters(), lr=config.world_lr, weight_decay=config.weight_decay
        )
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.actor_lr, weight_decay=config.weight_decay
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.critic_lr, weight_decay=config.weight_decay
        )

        self.replay_buffer = ReplayBuffer(config.replay_buffer_capacity)

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

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def reset_hidden_states(self):
        self.h = torch.zeros(1, 1, self.config.hidden_dim, device=device)

    def update_world_model(self):
        batch = self.replay_buffer.sample_batch(
            self.config.batch_size, self.config.seq_len
        )
        if batch[0] is None:
            return  # Not enough data to train

        obs_seq, act_seq, rew_seq, next_obs_seq, done_seq = batch
        act_seq = act_seq  # [batch_size, seq_len]
        rew_seq = rew_seq  # [batch_size, seq_len]

        outputs = self.world_model(obs_seq, act_seq, tau=self.temperature)
        recon_obs = outputs["recon_obs"]
        pred_reward = outputs["pred_reward"]
        kl_loss = outputs["kl_loss"]  # Shape: [batch_size]

        # Reconstruction loss
        if self.is_image:
            recon_loss = F.mse_loss(recon_obs, obs_seq[:, 1:], reduction='none')
            recon_loss = recon_loss.mean(dim=[1, 2, 3, 4])  # [batch_size]
        else:
            recon_loss = F.mse_loss(recon_obs, obs_seq[:, 1:], reduction='none')
            recon_loss = recon_loss.mean(dim=[1, 2])  # [batch_size]

        # Reward prediction loss
        reward_loss = F.mse_loss(
            pred_reward, symlog(rew_seq[:, 1:]), reduction='none'
        )
        reward_loss = reward_loss.mean(dim=1)  # [batch_size]

        # Total loss
        loss_world = recon_loss + reward_loss + kl_loss  # All shapes: [batch_size]
        loss_world = loss_world.mean()

        self.world_optimizer.zero_grad()
        loss_world.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.max_grad_norm)
        self.world_optimizer.step()

    def update_actor_and_critic(self):
        # Imagined rollouts
        batch = self.replay_buffer.sample_batch(self.config.batch_size, 1)
        if batch[0] is None:
            return  # Not enough data to train

        obs_seq, act_seq, _, _, _ = batch
        obs = obs_seq[:, 0]

        # Initialize hidden state with zeros
        imag_h = torch.zeros(self.config.batch_size, self.config.hidden_dim, device=device)

        if self.is_image:
            obs_encoded = self.world_model.obs_encoder(obs)
        else:
            obs_encoded = self.world_model.obs_encoder(obs)

        # Initialize imag_s with posterior sample from initial obs
        posterior_input = torch.cat([imag_h, obs_encoded], dim=-1)
        posterior_logits = self.world_model.posterior_net(posterior_input)
        posterior_logits = posterior_logits.view(
            self.config.batch_size, self.config.latent_dim, self.config.latent_categories
        )
        posterior_sample = gumbel_softmax(posterior_logits, tau=self.temperature, hard=False).view(
            self.config.batch_size, -1
        )
        imag_s = posterior_sample

        imag_states = []
        imag_rewards = []
        imag_values = []
        imag_action_probs = []
        imag_actions = []

        for _ in range(self.config.imagination_horizon):
            # Compute prior over latent state
            prior_logits = self.world_model.prior_net(imag_h)
            prior_logits = prior_logits.view(
                self.config.batch_size, self.config.latent_dim, self.config.latent_categories
            )
            imag_s = gumbel_softmax(prior_logits, tau=self.temperature, hard=False).view(self.config.batch_size, -1)

            # Get action probabilities from actor
            action_probs = self.actor(imag_h)
            action_dist = torch.distributions.Categorical(probs=action_probs)
            imag_action = action_dist.sample()
            imag_actions.append(imag_action)
            act_onehot = torch.nn.functional.one_hot(
                imag_action, num_classes=self.act_dim
            ).float()

            # Update imag_h using the RNN
            rnn_input = torch.cat([imag_s, act_onehot], dim=-1)
            imag_h, _ = self.world_model.rnn(
                rnn_input.unsqueeze(1), imag_h.unsqueeze(0)
            )
            imag_h = imag_h.squeeze(1)

            # Predict reward
            decoder_input = torch.cat([imag_h, imag_s], dim=-1)
            pred_reward = self.world_model.reward_decoder(decoder_input)
            imag_rewards.append(pred_reward.squeeze(-1))

            # Get value estimates from target critic
            q_values = self.target_critic(imag_h)
            imag_values.append(q_values)

            imag_states.append(imag_h)
            imag_action_probs.append(action_probs)

        # Convert lists to tensors
        imag_rewards = torch.stack(imag_rewards)  # [horizon, batch_size]
        imag_values = torch.stack(imag_values)  # [horizon, batch_size]
        imag_states = torch.stack(imag_states)  # [horizon, batch_size, hidden_dim]
        imag_action_probs = torch.stack(imag_action_probs)  # [horizon, batch_size, act_dim]
        imag_actions = torch.stack(imag_actions)  # [horizon, batch_size]

        # Append last value to imag_values for bootstrapping
        imag_values = torch.cat([imag_values, imag_values[-1:].detach()], dim=0)

        # Initialize tensors for advantages and returns
        advantages = torch.zeros_like(imag_rewards)
        returns = torch.zeros_like(imag_rewards)
        next_advantage = 0

        # Compute GAE (Generalized Advantage Estimation)
        for t in reversed(range(self.config.imagination_horizon)):
            delta = imag_rewards[t] + self.config.gamma * imag_values[t + 1] - imag_values[t]
            next_advantage = delta + self.config.gamma * self.config.lambda_ * next_advantage
            advantages[t] = next_advantage
            returns[t] = advantages[t] + imag_values[t]

        # Flatten tensors for loss computation
        imag_states_flat = imag_states.view(-1, self.config.hidden_dim)
        returns_flat = returns.view(-1)
        advantages_flat = advantages.view(-1)
        imag_action_probs_flat = imag_action_probs.view(-1, self.act_dim)
        imag_actions_flat = imag_actions.view(-1)

        # Critic update
        values = self.critic(imag_states_flat)
        critic_loss = F.mse_loss(values, returns_flat.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        # Actor update
        action_log_probs = torch.log(imag_action_probs_flat + 1e-8)
        selected_action_log_probs = action_log_probs.gather(1, imag_actions_flat.unsqueeze(1)).squeeze(1)
        actor_loss = - (advantages_flat.detach() * selected_action_log_probs)

        # Entropy regularization
        entropy = -torch.sum(
            imag_action_probs_flat * torch.log(imag_action_probs_flat + 1e-8), dim=-1
        )
        actor_loss -= self.config.entropy_scale * entropy

        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        # Update target critic
        self._soft_update(self.target_critic, self.critic)

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

        obs = torch.tensor(obs).float().to(device).unsqueeze(0)

        with torch.no_grad():
            if self.is_image:
                obs_encoded = self.world_model.obs_encoder(obs / 255.0)
            else:
                obs_encoded = self.world_model.obs_encoder(obs)

            h = self.h  # [1, 1, hidden_dim]

            # Compute posterior over latent variables
            posterior_input = torch.cat([h.squeeze(0), obs_encoded], dim=-1)
            posterior_logits = self.world_model.posterior_net(posterior_input)
            posterior_logits = posterior_logits.view(
                1, self.config.latent_dim, self.config.latent_categories
            )
            posterior_sample = gumbel_softmax(posterior_logits, tau=self.temperature, hard=False).view(1, -1)
            imag_s = posterior_sample  # [1, latent_dim * latent_categories]

            # Get action probabilities from actor
            action_probs = self.actor(h.squeeze(0))
            action_dist = torch.distributions.Categorical(probs=action_probs)
            action = action_dist.sample().cpu().numpy()[0]

            # Update hidden state
            act_onehot = torch.nn.functional.one_hot(
                torch.tensor([action], device=device), num_classes=self.act_dim
            ).float()

            rnn_input = torch.cat([imag_s, act_onehot], dim=-1)
            self.h, _ = self.world_model.rnn(rnn_input.unsqueeze(1), h)
            self.h = self.h.detach()  # Detach to prevent backprop through time

        return action

    def store_transition(self, obs, act, rew, next_obs, done):
        transition = (obs, act, rew, next_obs, done)
        self.replay_buffer.store(transition)

    def train(self, num_updates):
        for _ in range(num_updates):
            self.update_world_model()
            self.update_actor_and_critic()
            # Anneal temperature
            self.temperature = max(self.temperature * self.config.temperature_decay, self.config.min_temperature)


from tqdm import tqdm

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

    frame_idx = 0  # For temperature annealing
    avg_reward_window = 100  # Running average over the last 100 episodes

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
                        agent.train(num_updates=config.num_updates)

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

    return total_rewards



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--train_horizon", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--latent_categories", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--world_lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--imagination_horizon", type=int, default=15)
    parser.add_argument("--free_nats", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--replay_buffer_capacity", type=int, default=100000)
    parser.add_argument("--entropy_scale", type=float, default=0.001)
    parser.add_argument("--kl_balance_alpha", type=float, default=0.8)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=100)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--num_updates", type=int, default=5)
    parser.add_argument("--min_buffer_size", type=int, default=1000)
    parser.add_argument("--init_temperature", type=float, default=1.0)
    parser.add_argument("--temperature_decay", type=float, default=0.9995)
    parser.add_argument("--min_temperature", type=float, default=0.5)
    parser.add_argument("--actor_temperature", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.005)
    args = parser.parse_args()

    train_dreamer(args)
