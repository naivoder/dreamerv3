import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torchvision import transforms
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LATENT_DIM = 32  # Number of discrete latent variables
LATENT_CATEGORIES = 32  # Number of categories for each variable
HIDDEN_DIM = 256
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
WORLD_LR = 1e-4
GAMMA = 0.99
IMAGINATION_HORIZON = 15
FREE_NATS = 3.0  # For KL balancing
BATCH_SIZE = 32
SEQ_LEN = 50
REPLAY_BUFFER_CAPACITY = 100000
ENTROPY_SCALE = 1e-2  # For entropy regularization
KL_BALANCE_ALPHA = 0.8  # For KL balancing between forward and reverse KL
ALPHA = 0.6  # Prioritization exponent
BETA_START = 0.4  # Initial beta value for importance sampling
BETA_FRAMES = 100000  # Frames over which beta increases to 1


# Symlog functions
def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


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


# SumTree data structure for efficient sampling
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree structure
        self.data = [None] * capacity  # Stores actual transitions
        self.write = 0  # Points to the next index to write data

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data  # Store data in self.data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0  # Overwrite if capacity exceeded

    def update(self, idx, priority):
        change = priority - self.tree[idx]

        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]


# Prioritized Replay Buffer with SumTree
class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        capacity: maximum size of the buffer
        alpha: how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-6  # Small amount to avoid zero priority

    def store(self, transition):
        # Assign maximum priority to new transitions to ensure they are sampled at least once
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0
        self.tree.add(max_priority, transition)

    def sample_batch(self, batch_size, seq_len, beta=0.4):
        """
        batch_size: number of samples to draw
        seq_len: length of each sequence
        beta: controls how much importance sampling is used (0 - no corrections, 1 - full correction)
        """
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)

            # Collect sequences
            seq = []
            for offset in range(seq_len):
                idx_offset = idx + offset
                if idx_offset >= self.tree.capacity * 2 - 1:
                    break
                data_idx = idx_offset - self.tree.capacity + 1
                if data_idx >= self.capacity:
                    data_idx -= self.capacity
                seq_data = self.tree.data[data_idx]
                if seq_data is None:
                    break
                seq.append(seq_data)
            if len(seq) < seq_len:
                continue  # Skip if not enough sequence data
            batch.append(seq)
            idxs.append(idx)
            priorities.append(priority)

        if len(batch) == 0:
            return [None] * 7  # Not enough data

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.capacity * sampling_probabilities, -beta)
        is_weights /= is_weights.max()  # Normalize for stability

        # Unpack batch data
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = [], [], [], [], []
        for seq in batch:
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
            idxs,
            torch.tensor(is_weights, dtype=torch.float32).to(device),
        )

    def update_priorities(self, idxs, priorities):
        # Priorities should be positive
        priorities += self.epsilon
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority**self.alpha)


# Convolutional Encoder for Image Observations
class ConvEncoder(nn.Module):
    def __init__(self, obs_shape):
        super(ConvEncoder, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(
                obs_shape[0], 32, kernel_size=8, stride=4
            ),  # Output: (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Output: (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # Output: (128, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, HIDDEN_DIM),
            nn.ReLU(),
        )

    def forward(self, obs):
        obs = obs / 255.0  # Normalize pixel values
        return self.conv_net(obs)


# Convolutional Decoder for Reconstructing Image Observations
class ConvDecoder(nn.Module):
    def __init__(self, obs_shape):
        super(ConvDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM + LATENT_DIM * LATENT_CATEGORIES, 128 * 7 * 7),
            nn.ReLU(),
        )
        self.deconv_net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),  # Output: (64, 9, 9)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),  # Output: (32, 20, 20)
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, obs_shape[0], kernel_size=8, stride=4, output_padding=0
            ),  # Output: (channels, 84, 84)
            nn.Sigmoid(),  # Output pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 7, 7)
        return self.deconv_net(x)


# MLP Encoder for Vector Observations
class MLPEncoder(nn.Module):
    def __init__(self, obs_dim):
        super(MLPEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs)


# MLP Decoder for Reconstructing Vector Observations
class MLPDecoder(nn.Module):
    def __init__(self, obs_dim):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(HIDDEN_DIM + LATENT_DIM * LATENT_CATEGORIES, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, obs_dim),
        )

    def forward(self, x):
        return self.net(x)


# World Model with Discrete Latent Representations and KL Balancing
class WorldModel(nn.Module):
    def __init__(self, obs_shape, act_dim, is_image):
        super(WorldModel, self).__init__()
        self.is_image = is_image
        self.act_dim = act_dim

        if self.is_image:
            self.obs_encoder = ConvEncoder(obs_shape)
            self.obs_decoder = ConvDecoder(obs_shape)
            self.obs_shape = obs_shape
            self.obs_dim = None
        else:
            self.obs_dim = obs_shape[0]
            self.obs_encoder = MLPEncoder(self.obs_dim)
            self.obs_decoder = MLPDecoder(self.obs_dim)
            self.obs_shape = (self.obs_dim,)
        self.rnn = nn.GRU(HIDDEN_DIM + act_dim, HIDDEN_DIM, batch_first=True)
        self.prior_net = nn.Linear(HIDDEN_DIM, LATENT_DIM * LATENT_CATEGORIES)
        self.posterior_net = nn.Linear(
            HIDDEN_DIM + HIDDEN_DIM, LATENT_DIM * LATENT_CATEGORIES
        )
        self.reward_decoder = nn.Linear(HIDDEN_DIM + LATENT_DIM * LATENT_CATEGORIES, 1)

    def forward(self, obs_seq, act_seq, h=None):
        batch_size, seq_len = obs_seq.size(0), obs_seq.size(1)

        if self.is_image:
            obs_seq = obs_seq.view(
                batch_size * seq_len, *self.obs_shape
            )  # [batch_size * seq_len, c, h, w]
            obs_encoded = self.obs_encoder(obs_seq)
        else:
            obs_seq = obs_seq.view(batch_size * seq_len, -1)  # [batch_size * seq_len, obs_dim]
            obs_encoded = self.obs_encoder(obs_seq)

        obs_encoded = obs_encoded.view(batch_size, seq_len, -1)

        act_seq_onehot = torch.nn.functional.one_hot(
            act_seq, num_classes=self.act_dim
        ).float()
        rnn_input = torch.cat([obs_encoded[:, :-1], act_seq_onehot[:, :-1]], dim=-1)
        if h is None:
            h = torch.zeros(1, batch_size, HIDDEN_DIM, device=obs_seq.device)
        rnn_output, _ = self.rnn(rnn_input, h)
        rnn_output = rnn_output  # [batch_size, seq_len - 1, HIDDEN_DIM]

        prior_logits = self.prior_net(rnn_output)
        posterior_input = torch.cat([rnn_output, obs_encoded[:, 1:]], dim=-1)
        posterior_logits = self.posterior_net(posterior_input)

        prior_logits = prior_logits.view(
            batch_size, seq_len - 1, LATENT_DIM, LATENT_CATEGORIES
        )
        posterior_logits = posterior_logits.view(
            batch_size, seq_len - 1, LATENT_DIM, LATENT_CATEGORIES
        )

        # Sample from posterior
        posterior_sample = gumbel_softmax(posterior_logits, hard=True).view(
            batch_size, seq_len - 1, -1
        )

        # Compute KL divergence with KL balancing
        kl_loss = self.compute_kl_loss(prior_logits, posterior_logits)

        # Decode observation and reward
        decoder_input = torch.cat([rnn_output, posterior_sample], dim=-1)
        if self.is_image:
            recon_obs = self.obs_decoder(decoder_input.view(-1, decoder_input.size(-1)))
            recon_obs = recon_obs.view(batch_size, seq_len - 1, *self.obs_shape)
        else:
            recon_obs = self.obs_decoder(decoder_input.view(-1, decoder_input.size(-1)))
            recon_obs = recon_obs.view(batch_size, seq_len - 1, *self.obs_shape)
        pred_reward = self.reward_decoder(decoder_input)

        outputs = {
            "recon_obs": recon_obs,
            "pred_reward": pred_reward.squeeze(-1),
            "kl_loss": kl_loss,
            "rnn_h": rnn_output,
            "posterior_sample": posterior_sample,
        }
        return outputs

    def compute_kl_loss(self, prior_logits, posterior_logits):
        prior_dist = torch.distributions.Categorical(logits=prior_logits)
        posterior_dist = torch.distributions.Categorical(logits=posterior_logits)

        kl_div_forward = torch.distributions.kl_divergence(
            posterior_dist, prior_dist
        ).sum(-1)
        kl_div_reverse = torch.distributions.kl_divergence(
            prior_dist, posterior_dist
        ).sum(-1)

        kl_loss = (
            KL_BALANCE_ALPHA * kl_div_forward + (1 - KL_BALANCE_ALPHA) * kl_div_reverse
        )
        kl_loss = torch.clamp(kl_loss, min=FREE_NATS).mean()
        return kl_loss


# Actor Network for Discrete Actions with Entropy Regularization
class Actor(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, act_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        action_probs = torch.softmax(logits, dim=-1)
        return action_probs


# Critic Network with Twohot Encoding
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(
                HIDDEN_DIM, 51
            ),  # Using 51 atoms for distributional value estimation
        )
        self.v_min = -10
        self.v_max = 10
        self.n_atoms = 51
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(device)

    def forward(self, x):
        logits = self.net(x)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities

    def get_q_values(self, probabilities):
        q_values = torch.sum(probabilities * self.support, dim=-1)
        return q_values


# DreamerV3 Agent
class DreamerV3:
    def __init__(self, obs_shape, act_dim, is_image):
        self.obs_shape = obs_shape
        self.act_dim = act_dim
        self.is_image = is_image

        self.world_model = WorldModel(obs_shape, act_dim, is_image).to(device)
        self.actor = Actor(HIDDEN_DIM, act_dim).to(device)
        self.critic = Critic(HIDDEN_DIM).to(device)
        self.target_critic = Critic(HIDDEN_DIM).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=WORLD_LR)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY, alpha=ALPHA)

        # Hidden states
        self.h = None
        self.reset_hidden_states()
        self.beta = BETA_START  # Importance sampling exponent

    def reset_hidden_states(self):
        self.h = torch.zeros(1, 1, HIDDEN_DIM, device=device)

    def update_world_model(self):
        batch = self.replay_buffer.sample_batch(BATCH_SIZE, SEQ_LEN, beta=self.beta)
        if batch[0] is None:
            return  # Not enough data to train

        obs_seq, act_seq, rew_seq, next_obs_seq, done_seq, idxs, is_weights = batch
        act_seq = act_seq  # [batch_size, seq_len]
        rew_seq = rew_seq  # [batch_size, seq_len]
        is_weights = is_weights.unsqueeze(1)  # [batch_size, 1]

        outputs = self.world_model(obs_seq, act_seq)
        recon_obs = outputs["recon_obs"]
        pred_reward = outputs["pred_reward"]
        kl_loss = outputs["kl_loss"]

        # Reconstruction loss
        if self.is_image:
            recon_loss = nn.MSELoss(reduction="none")(recon_obs, obs_seq[:, 1:])
            recon_loss = recon_loss.mean(dim=[1, 2, 3, 4])  # [batch_size]
        else:
            recon_loss = nn.MSELoss(reduction="none")(recon_obs, obs_seq[:, 1:])
            recon_loss = recon_loss.mean(dim=[1, 2])  # [batch_size]

        # Reward prediction loss
        reward_loss = nn.MSELoss(reduction="none")(pred_reward, symlog(rew_seq[:, 1:]))
        reward_loss = reward_loss.mean(dim=1)  # [batch_size]

        # Total loss
        loss_world = recon_loss + reward_loss + kl_loss  # [batch_size]

        # Apply importance sampling weights
        weighted_loss_world = (loss_world * is_weights.squeeze(1)).mean()

        self.world_optimizer.zero_grad()
        weighted_loss_world.backward()
        self.world_optimizer.step()

        # Update priorities
        priorities = loss_world.detach().cpu().numpy()
        self.replay_buffer.update_priorities(idxs, priorities)

    def update_actor_and_critic(self):
        # Imagined rollouts
        batch = self.replay_buffer.sample_batch(BATCH_SIZE, 1, beta=self.beta)
        if batch[0] is None:
            return  # Not enough data to train

        obs_seq, act_seq, _, _, _, idxs, is_weights = batch
        obs = obs_seq[:, 0]
        act = act_seq[:, 0]
        is_weights = is_weights  # [batch_size]

        h = torch.zeros(1, BATCH_SIZE, HIDDEN_DIM, device=device)
        if self.is_image:
            obs_encoded = self.world_model.obs_encoder(obs)
        else:
            obs_encoded = self.world_model.obs_encoder(obs)

        h = h.permute(1, 0, 2)  # [batch_size, HIDDEN_DIM]
        h = h[:, -1]  # [batch_size, HIDDEN_DIM]

        imag_h = h
        imag_s = None  # Initialize latent state

        imag_states = []
        imag_rewards = []
        imag_values = []
        imag_action_probs = []

        for _ in range(IMAGINATION_HORIZON):
            action_probs = self.actor(imag_h)
            action_dist = torch.distributions.Categorical(probs=action_probs)
            imag_action = action_dist.sample()

            act_onehot = torch.nn.functional.one_hot(
                imag_action, num_classes=self.act_dim
            ).float()

            rnn_input = torch.cat([obs_encoded, act_onehot], dim=-1)
            imag_h, _ = self.world_model.rnn(
                rnn_input.unsqueeze(1), imag_h.unsqueeze(0)
            )
            imag_h = imag_h.squeeze(1)

            prior_logits = self.world_model.prior_net(imag_h)
            prior_logits = prior_logits.view(BATCH_SIZE, LATENT_DIM, LATENT_CATEGORIES)
            prior_sample = gumbel_softmax(prior_logits, hard=True).view(BATCH_SIZE, -1)
            imag_s = prior_sample

            decoder_input = torch.cat([imag_h, imag_s], dim=-1)
            pred_reward = self.world_model.reward_decoder(decoder_input)
            imag_rewards.append(pred_reward.squeeze(-1))

            # Get value estimates from target critic
            probs = self.target_critic(imag_h)
            q_values = self.target_critic.get_q_values(probs)
            imag_values.append(q_values)

            imag_states.append(imag_h)
            imag_action_probs.append(action_probs)

        # Compute returns and advantages
        imag_rewards = torch.stack(imag_rewards)  # [horizon, batch_size]
        imag_values = torch.stack(imag_values)  # [horizon, batch_size]

        returns = torch.zeros_like(imag_rewards)
        next_value = torch.zeros(BATCH_SIZE, device=device)

        for t in reversed(range(IMAGINATION_HORIZON)):
            next_value = imag_rewards[t] + GAMMA * next_value
            returns[t] = next_value

        # Flatten tensors for loss computation
        imag_states_flat = torch.cat(
            imag_states, dim=0
        )  # [horizon * batch_size, HIDDEN_DIM]
        returns_flat = returns.view(-1)
        is_weights_flat = is_weights.repeat(IMAGINATION_HORIZON)

        # Critic update with Twohot Encoding
        probs = self.critic(imag_states_flat)
        q_values = self.critic.get_q_values(probs)
        target_probs = self._compute_target_probs(returns_flat)
        critic_loss = -torch.sum(target_probs * torch.log(probs + 1e-8), dim=-1)
        weighted_critic_loss = critic_loss * is_weights_flat

        # Actor update with Entropy Regularization
        imag_action_probs_flat = torch.cat(imag_action_probs, dim=0)
        actor_loss = -q_values  # No detach here
        entropy = -torch.sum(
            imag_action_probs_flat * torch.log(imag_action_probs_flat + 1e-8), dim=-1
        )
        actor_loss += ENTROPY_SCALE * entropy
        weighted_actor_loss = actor_loss * is_weights_flat

        # Combine losses
        total_loss = (weighted_actor_loss + weighted_critic_loss).mean()

        # Backpropagate
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Update priorities based on TD-errors
        with torch.no_grad():
            td_errors = returns_flat - q_values
            td_errors_batch = td_errors.view(IMAGINATION_HORIZON, BATCH_SIZE).mean(
                dim=0
            )
            priorities = td_errors_batch.abs().cpu().numpy()
            self.replay_buffer.update_priorities(idxs, priorities)

        # Update target critic
        self._soft_update(self.target_critic, self.critic)

    def _compute_target_probs(self, returns):
        batch_size = returns.size(0)
        target_probs = torch.zeros(batch_size, self.critic.n_atoms, device=device)
        b = (returns - self.critic.v_min) / self.critic.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        l = torch.clamp(l, 0, self.critic.n_atoms - 1)
        u = torch.clamp(u, 0, self.critic.n_atoms - 1)

        d_m_u = u.float() - b
        d_b_l = b - l.float()

        target_probs[range(batch_size), l] += d_m_u
        target_probs[range(batch_size), u] += d_b_l
        return target_probs

    def _soft_update(self, target, source, tau=0.005):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def act(self, obs, reset=False):
        if reset:
            self.reset_hidden_states()

        obs = torch.tensor(obs).float().to(device).unsqueeze(0)

        with torch.no_grad():
            if self.is_image:
                obs_encoded = self.world_model.obs_encoder(obs / 255.0)
            else:
                obs_encoded = self.world_model.obs_encoder(obs)

            action_probs = self.actor(self.h[-1])
            action_dist = torch.distributions.Categorical(probs=action_probs)
            action = action_dist.sample().cpu().numpy()[0]

            # Update hidden state
            act_onehot = torch.nn.functional.one_hot(
                torch.tensor([action], device=device), num_classes=self.act_dim
            ).float()
            rnn_input = torch.cat([obs_encoded, act_onehot], dim=-1)
            self.h, _ = self.world_model.rnn(rnn_input.unsqueeze(1), self.h)
            self.h = self.h.detach()  # Detach to prevent backprop through time

        return action

    def store_transition(self, obs, act, rew, next_obs, done):
        transition = (obs, act, rew, next_obs, done)
        self.replay_buffer.store(transition)

    def train(self, num_updates):
        for _ in range(num_updates):
            self.update_world_model()
            self.update_actor_and_critic()


# Training Loop
def train_dreamer(args):
    env = gym.make(args.env)
    # Determine if the observation space is image-based or vector-based
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3:
        is_image = True
    else:
        is_image = False

    if is_image:
        # Preprocess observations to shape (4, 84, 84)
        transform = transforms.Compose(
            [
                transforms.Resize((84, 84)),
                transforms.Grayscale(num_output_channels=1),
            ]
        )
        obs_shape = (4, 84, 84)
    else:
        obs_shape = obs_space.shape
        transform = None

    act_dim = env.action_space.n

    agent = DreamerV3(obs_shape, act_dim, is_image)
    total_rewards = []

    frame_idx = 0  # For beta annealing

    for episode in range(args.episodes):
        obs, _ = env.reset()
        if is_image:
            obs = transform(torch.tensor(obs).permute(2, 0, 1)).numpy()
            obs = np.repeat(obs, 4, axis=0)  # Stack 4 frames
            frame_stack = deque([obs.copy() for _ in range(4)], maxlen=4)
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
                next_obs_processed = transform(
                    torch.tensor(next_obs).permute(2, 0, 1)
                ).numpy()
                frame_stack.append(next_obs_processed)
                next_obs = np.concatenate(frame_stack, axis=0)  # Update frame stack
                agent.store_transition(obs, action, reward, next_obs, done)
                obs = next_obs
            else:
                next_obs = next_obs.astype(np.float32)
                agent.store_transition(obs, action, reward, next_obs, done)
                obs = next_obs

            episode_reward += reward

            frame_idx += 1
            agent.beta = min(
                1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES
            )

            if len(agent.replay_buffer.tree.data) >= BATCH_SIZE * SEQ_LEN:
                agent.train(num_updates=1)

            if done:
                total_rewards.append(episode_reward)
                print(f"Episode {episode} Reward: {episode_reward}")
                break  # Exit the while loop when the episode is done

    return total_rewards


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    train_dreamer(args)
