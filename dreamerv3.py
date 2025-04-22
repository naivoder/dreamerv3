import os
import cv2
import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import imageio
from collections import deque
from torch.distributions import Categorical, Independent
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.simplefilter("ignore")

gym.register_envs(ale_py)

def preprocess(image):
    return image.astype(np.float32) / 255.0

def quantize(image):
    return (image * 255).clip(0, 255).astype(np.uint8)

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Config:
    def __init__(self, args):
        self.capacity = 1_000_000         # Increase buffer capacity on larger machine...
        self.batch_size = 16             # 16 or more on larger machine...               
        self.sequence_length = 8       # 64 on larger machine     
        self.embed_dim = 1024
        self.latent_dim = 32
        self.num_classes = 32
        self.deter_dim = 512             
        self.lr = 6e-4
        self.eps = 1e-5                 
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.discount = 0.99
        self.kl_scale = 0.1
        self.imagination_horizon = 15       # 15 is correct 
        self.min_buffer_size = 5000        
        self.episodes = args.episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.free_bits = 1.0            
        self.entropy_coef = 0.001         
        self.updates_per_step = 1       # 5 on larger machine...        
        self.grad_clip = 100.0            

class ReplayBuffer:
    def __init__(self, config, device, obs_shape):
        self.capacity = config.capacity
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.device = device
        self.obs_shape = obs_shape
        
        self.obs_buf = np.zeros((config.capacity, *obs_shape), dtype=np.uint8)
        self.act_buf = np.zeros(config.capacity, dtype=np.int64)
        self.rew_buf = np.zeros(config.capacity, dtype=np.float32)
        self.done_buf = np.zeros(config.capacity, dtype=np.bool_)
        self.pos = 0
        self.full = False

    def store(self, obs, act, rew, done):
        idx = self.pos % self.capacity
        self.obs_buf[idx] = obs
        self.act_buf[idx] = act
        self.rew_buf[idx] = rew
        self.done_buf[idx] = done
        self.pos += 1
        if self.pos >= self.capacity:
            self.full = True
            self.pos = 0

    def sample(self):
        current_size = self.capacity if self.full else self.pos
        valid_end = current_size - self.sequence_length
        
        start_indices = np.random.randint(0, valid_end, size=self.batch_size)
        indices = (start_indices[:, None] + np.arange(self.sequence_length)) % self.capacity
        
        obs = torch.as_tensor(self.obs_buf[indices], dtype=torch.float32, device=self.device)
        obs = obs.div_(255.0).permute(1, 0, 2, 3, 4)
        
        return {
            "observation": obs,
            "action": torch.as_tensor(self.act_buf[indices], dtype=torch.long, device=self.device).permute(1, 0),
            "reward": torch.as_tensor(self.rew_buf[indices], dtype=torch.float32, device=self.device).permute(1, 0),
            "done": torch.as_tensor(self.done_buf[indices], dtype=torch.float32, device=self.device).permute(1, 0),
        }

    def __len__(self):
        return self.capacity if self.full else self.pos

class OneHotCategoricalStraightThrough(Categorical):
    def sample(self, sample_shape=torch.Size()):
        samples = super().sample(sample_shape)
        probs = self.probs
        while len(probs.shape) < len(samples.shape):
            probs = probs.unsqueeze(0)
        return probs + (samples - probs).detach()
    

class ObservationEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, embed_dim)
        self.apply(init_weights)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ObservationDecoder(nn.Module):
    def __init__(self, feature_dim, out_channels, output_size):
        super().__init__()
        self.feature_dim = feature_dim
        self.out_channels = out_channels
        self.output_size = output_size
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, out_channels * 255, kernel_size=1)
        )
        
        self.fc = nn.Linear(feature_dim, 256)
        self.apply(init_weights)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Project to initial deconv dimensions
        x = self.fc(x)
        x = x.view(batch_size, 256, 1, 1)
        
        # Apply deconvolution layers
        x = self.deconv(x)  # Output shape: [B, out_channels*255, 64, 64]
        
        # Reshape and permute for categorical distribution
        x = x.view(batch_size, self.out_channels, 255, *self.output_size)
        x = x.permute(0, 1, 3, 4, 2)  # [B, C, H, W, 255]
        
        return Independent(
            OneHotCategoricalStraightThrough(logits=x),
            reinterpreted_batch_ndims=3
        )

class TwoHotCategoricalStraightThrough(torch.distributions.Distribution):
    def __init__(self, logits, bins=255, low=-20.0, high=20.0):
        super().__init__(validate_args=False)
        self.bins = bins
        self.low = low
        self.high = high
        self.logits = logits
        self.bin_size = (high - low) / (bins - 1)
        self.bin_centers = torch.linspace(
            low, high, bins, 
            device=logits.device,
            dtype=logits.dtype
        )
        
    def log_prob(self, value):
        # Symlog and normalize value to [low, high] range
        value = symlog(value)
        value = torch.clamp(value, self.low, self.high)
        
        # Convert to bin indices
        normalized = (value - self.low) / self.bin_size
        bin_idx = normalized.floor().long()
        bin_next = bin_idx + 1
        alpha = normalized - bin_idx
        
        # Create two-hot distribution
        log_probs = F.log_softmax(self.logits, dim=-1)
        bin_probs = log_probs.gather(-1, bin_idx.clamp(0, self.bins-1))
        next_probs = log_probs.gather(-1, bin_next.clamp(0, self.bins-1))
        
        return (1 - alpha) * bin_probs + alpha * next_probs
    
    def sample(self, sample_shape=torch.Size()):
        probs = F.softmax(self.logits, dim=-1)
        samples = torch.multinomial(probs, 1).float()
        # Straight-through gradient
        return probs + (samples - probs).detach()
    
    @property
    def mean(self):
        probs = F.softmax(self.logits, dim=-1)
        return symexp((probs * self.bin_centers).sum(dim=-1, keepdim=True))
    
    @property
    def mode(self):
        probs = F.softmax(self.logits, dim=-1)
        max_idx = torch.argmax(probs, dim=-1)
        return symexp(self.bin_centers[max_idx])

class RewardDecoder(nn.Module):
    def __init__(self, in_dim, bins=255):
        super().__init__()
        self.net = nn.Linear(in_dim, bins)
        self.bins = bins
        self.apply(init_weights)

    def forward(self, x):
        logits = self.net(x)
        return TwoHotCategoricalStraightThrough(logits=logits)

class RSSM(nn.Module):
    def __init__(self, action_dim, latent_dim, num_classes, deter_dim, embed_dim):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.deter_dim = deter_dim
        self.embed_dim = embed_dim
        self.gru = nn.GRUCell(latent_dim * num_classes + action_dim, deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, 200),
            nn.ReLU(),
            nn.Linear(200, latent_dim * num_classes),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, 200),
            nn.ReLU(),
            nn.Linear(200, latent_dim * num_classes),
        )
        self.apply(init_weights)

    def init_state(self, batch_size, device):
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.latent_dim, self.num_classes, device=device)
        stoch[:, :, 0] = 1.0
        return (stoch, deter)

    def observe(self, embed_seq, action_seq, init_state):
        T, B = action_seq.shape[:2]
        priors, posteriors, features = [], [], []
        stoch, deter = init_state
        for t in range(T):
            stoch_flat = stoch.view(B, -1)
            x = torch.cat([stoch_flat, action_seq[t]], dim=-1)
            deter = self.gru(x, deter)
            
            # Prior only uses deterministic state
            prior_logits = self.prior_net(deter).view(B, self.latent_dim, self.num_classes)
            prior_dist = Independent(OneHotCategoricalStraightThrough(logits=prior_logits), 1)
            
            # Posterior uses both deter and embedding
            post_input = torch.cat([deter, embed_seq[t]], dim=-1)
            posterior_logits = self.posterior_net(post_input).view(B, self.latent_dim, self.num_classes)
            posterior_dist = Independent(OneHotCategoricalStraightThrough(logits=posterior_logits), 1)
            
            # Straight-through sampling
            stoch = F.gumbel_softmax(posterior_logits, tau=1.0, hard=True)
            feature = torch.cat([deter, stoch.view(B, -1)], dim=-1)
            
            features.append(feature)
            priors.append(prior_dist)
            posteriors.append(posterior_dist)
        
        return (priors, posteriors), torch.stack(features, dim=0)

    def imagine(self, init_state, actor, horizon):
        stoch, deter = init_state
        features, actions = [], []
        B = deter.size(0)
        for _ in range(horizon):
            feature = torch.cat([deter, stoch.view(B, -1)], dim=-1)
            features.append(feature)
            
            with torch.no_grad():
                action_dist = actor(feature)
                action = action_dist.sample()
                actions.append(action)
                action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
            
            stoch_flat = stoch.view(B, -1)
            x = torch.cat([stoch_flat, action_onehot], dim=-1)
            deter = self.gru(x, deter)
            
            prior_logits = self.prior_net(deter).view(B, self.latent_dim, self.num_classes)
            stoch = F.gumbel_softmax(prior_logits, tau=1.0, hard=True)
        
        return torch.stack(features, dim=0), torch.stack(actions, dim=0)

class WorldModel(nn.Module):
    def __init__(self, in_channels, action_dim, embed_dim, latent_dim, num_classes, deter_dim, obs_size, lr=1e-4, eps=1e-5):
        super().__init__()
        self.encoder = ObservationEncoder(in_channels, embed_dim)
        self.rssm = RSSM(action_dim, latent_dim, num_classes, deter_dim, embed_dim)
        feature_dim = deter_dim + latent_dim * num_classes
        self.decoder = ObservationDecoder(feature_dim, in_channels, obs_size)
        self.reward_decoder = RewardDecoder(feature_dim)
        self.continue_decoder = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=eps)

    def observe(self, observations, actions):
        T, B = observations.shape[:2]
        obs_flat = observations.reshape(T * B, *observations.shape[2:])

        embed = self.encoder(obs_flat).view(T, B, -1)
        actions_onehot = F.one_hot(actions.long(), num_classes=self.rssm.action_dim).float()
        
        init_state = self.rssm.init_state(B, observations.device)
        (priors, posteriors), features = self.rssm.observe(embed, actions_onehot, init_state)
        
        feat_dim = features.size(-1)
        features_flat = features.view(T * B, feat_dim)
        
        recon_dist = self.decoder(features_flat)
        obs_target = (observations * 255).long().reshape(-1, *observations.shape[2:])
        
        reward_dist = self.reward_decoder(features_flat)
        continue_pred = self.continue_decoder(features_flat)
        
        return (priors, posteriors), features, recon_dist, reward_dist, continue_pred, obs_target

    def imagine(self, init_state, actor, horizon):
        features, actions = self.rssm.imagine(init_state, actor, horizon)
        T, B, feat_dim = features.shape
        features_flat = features.view(T * B, feat_dim)
        
        reward_dist = self.reward_decoder(features_flat)
        continue_pred = self.continue_decoder(features_flat)
        
        return features, actions, reward_dist, continue_pred

    def decode(self, features):
        return self.decoder(features)

class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, action_dim))
        self.apply(init_weights)

    def forward(self, x):
        logits = self.net(x)
        return Categorical(logits=logits)

class Critic(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, 1))
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)

class DreamerV3:
    def __init__(self, obs_shape, action_dim, config):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.config = config
        self.replay_buffer = ReplayBuffer(config, config.device, obs_shape)
        self.device = config.device
        
        self.world_model = WorldModel(
            obs_shape[0], action_dim, config.embed_dim, config.latent_dim,
            config.num_classes, config.deter_dim, obs_shape[1:], 
            lr=config.lr, eps=config.eps
        ).to(self.device)
        
        feature_dim = config.deter_dim + config.latent_dim * config.num_classes
        self.actor = Actor(feature_dim, action_dim).to(self.device)
        self.critic = Critic(feature_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        self.hidden_state = None
        self.step = 0

    def init_hidden_state(self):
        self.hidden_state = self.world_model.rssm.init_state(1, self.device)

    def act(self, observation):
        obs = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.hidden_state is None:
                self.init_hidden_state()
            
            stoch, deter = self.hidden_state
            embed = self.world_model.encoder(obs)
            
            # Posterior update
            post_input = torch.cat([deter, embed], dim=-1)
            posterior_logits = self.world_model.rssm.posterior_net(post_input)
            posterior_logits = posterior_logits.view(1, self.world_model.rssm.latent_dim, 
                                                   self.world_model.rssm.num_classes)
            stoch = F.gumbel_softmax(posterior_logits, tau=1.0, hard=True)
            
            feature = torch.cat([deter, stoch.view(1, -1)], dim=-1)
            action_dist = self.actor(feature)
            action = action_dist.sample()
            
            # Prior update
            action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
            stoch_flat = stoch.view(1, -1)
            x = torch.cat([stoch_flat, action_onehot], dim=-1)
            deter = self.world_model.rssm.gru(x, deter)
            
            prior_logits = self.world_model.rssm.prior_net(deter)
            prior_logits = prior_logits.view(1, self.world_model.rssm.latent_dim, 
                                           self.world_model.rssm.num_classes)
            
            self.hidden_state = (stoch, deter)
        
        return int(action.item())

    def store_transition(self, obs, action, reward, done):
        self.replay_buffer.store(obs, action, reward, done)

    def update_world_model(self, batch):
        obs = batch["observation"].permute(1, 0, 2, 3, 4)  
        actions = batch["action"].permute(1, 0)  
        rewards = batch["reward"].permute(1, 0)  
        dones = batch["done"].permute(1, 0)  
        
        (priors, posteriors), features, recon_dist, reward_dist, continue_pred, obs_target = \
            self.world_model.observe(obs, actions)
        
        # Reconstruction loss (image)
        recon_loss = -recon_dist.log_prob(obs_target).mean()
        
        # Reward loss (two-hot)
        reward_loss = -reward_dist.log_prob(rewards.reshape(-1, 1)).mean()
        
        # Continue loss
        continue_loss = F.binary_cross_entropy(continue_pred, (1 - dones).reshape(-1, 1))
        
        # KL loss with free bits
        kl_loss = 0
        free_bits = torch.tensor(self.config.free_bits, device=self.device)
        for prior, posterior in zip(priors, posteriors):
            kl_t = torch.distributions.kl_divergence(posterior, prior)
            kl_t = torch.mean(kl_t, dim=0)  # Average over batch, keep latents
            kl_t = torch.sum(torch.clamp(kl_t, min=free_bits))
            kl_loss += kl_t
        kl_loss /= len(priors)
        
        total_loss = recon_loss + reward_loss + continue_loss + self.config.kl_scale * kl_loss
        
        self.world_model.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.grad_clip)
        self.world_model.optimizer.step()
        
        return {
            "world_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item(),
            "kl_loss": kl_loss.item()
        }

    def update_actor_and_critic(self, init_state):
        horizon = self.config.imagination_horizon
        with torch.no_grad():
            features, actions, reward_dist, continue_pred = \
                self.world_model.imagine(init_state, self.actor, horizon)
            
            rewards = reward_dist.mean
            continues = continue_pred
            discounts = self.config.discount * continues
        
        # Calculate lambda returns
        T, B, feat_dim = features.shape
        features_flat = features.reshape(-1, feat_dim)
        values = self.critic(features_flat).reshape(T, B)
        
        returns = torch.zeros_like(values)
        last = values[-1]
        for t in reversed(range(T)):
            last = returns[t] = rewards[t] + discounts[t] * last
        
        # Critic update
        critic_loss = F.mse_loss(values, returns.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.critic_optimizer.step()
        
        # Actor update
        with torch.no_grad():
            advantages = returns - values
        
        action_dist = self.actor(features_flat)
        log_probs = action_dist.log_prob(actions.reshape(-1))
        entropy = action_dist.entropy().mean()
        
        actor_loss = -(log_probs * advantages.reshape(-1).detach()).mean() - self.config.entropy_coef * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.actor_optimizer.step()
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_entropy": entropy.item()
        }

    def train(self):
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None
        
        losses = {
            "world_loss": 0, "recon_loss": 0, "reward_loss": 0,
            "continue_loss": 0, "kl_loss": 0,
            "actor_loss": 0, "critic_loss": 0, "actor_entropy": 0
        }
        
        for _ in range(self.config.updates_per_step):
            batch = self.replay_buffer.sample()
            
            # World model update
            wm_losses = self.update_world_model(batch)
            for k, v in wm_losses.items():
                losses[k] += v / self.config.updates_per_step
            
            # Actor-critic update
            B = batch["observation"].shape[1]
            init_state = self.world_model.rssm.init_state(B, self.device)
            ac_losses = self.update_actor_and_critic(init_state)
            for k, v in ac_losses.items():
                losses[k] += v / self.config.updates_per_step
        
        self.step += 1
        return losses

    def save_checkpoint(self, env_name):
        os.makedirs("weights", exist_ok=True)
        torch.save({
            'world_model': self.world_model.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'step': self.step,
        }, f"weights/{env_name}_dreamerv3.pt")

    def load_checkpoint(self, env_name):
        checkpoint = torch.load(f"weights/{env_name}_dreamerv3.pt")
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.step = checkpoint['step']


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env, repeat=4, clip_reward=True, no_ops=0, fire_first=False):
        super().__init__(env)
        self.repeat = repeat
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first
        self.frame_buffer = np.zeros((2, *env.observation_space.shape), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        term, trunc = False, False
        for i in range(self.repeat):
            state, reward, term, trunc, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(reward, -1, 1)
            total_reward += reward
            self.frame_buffer[i % 2] = state
            if term or trunc:
                break
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, total_reward, term, trunc, info

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, term, trunc, info = self.env.step(0)
            if term or trunc:
                state, info = self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == "FIRE"
            state, _, term, trunc, info = self.env.step(1)
        self.frame_buffer = np.zeros((2, *self.env.observation_space.shape), dtype=np.float32)
        self.frame_buffer[0] = state
        return state, info

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape=(64, 64)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(0.0, 1.0, self.shape, dtype=np.float32)

    def observation(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, self.shape, interpolation=cv2.INTER_AREA)
        return preprocess(state)

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, size=4):
        super().__init__(env)
        self.size = int(size)
        self.stack = deque([], maxlen=self.size)
        shape = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.size, *shape), dtype=np.float32)

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        self.stack = deque([state] * self.size, maxlen=self.size)
        return np.array(self.stack), info

    def observation(self, state):
        self.stack.append(state)
        return np.array(self.stack)

class AtariEnv:
    def __init__(self, env_id, shape=(64, 64), repeat=4, clip_rewards=False, no_ops=0, fire_first=False):
        base_env = gym.make(env_id, render_mode="rgb_array")
        env = RepeatActionAndMaxFrame(base_env, repeat, clip_rewards, no_ops, fire_first)
        env = PreprocessFrame(env, shape)
        env = StackFrames(env, repeat)
        self.env = env

    def make(self):
        return self.env

def save_animation(frames, filename):
    with imageio.get_writer(filename, mode="I", loop=0) as writer:
        for frame in frames:
            writer.append_data(frame)

def create_animation(env_name, agent, seeds=5):
    env = AtariEnv("ALE/"+env_name, shape=(64, 64), repeat=4, clip_rewards=False).make()
    save_prefix = env_name.split("/")[-1]
    agent.load_checkpoint(save_prefix)
    best_total_reward, best_frames = float("-inf"), None
    for _ in range(seeds):
        state, info = env.reset()
        frames, total_reward = [], 0
        term, trunc = False, False
        while not (term or trunc):
            frames.append(env.render())
            action = agent.act(state)
            next_state, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            state = next_state
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_frames = frames
    save_animation(best_frames, f"environments/{save_prefix}.gif")

def train_dreamer(args):
    env = AtariEnv(args.env).make()
    obs_shape = env.observation_space.shape
    act_dim = env.action_space.n
    save_prefix = args.env.split("/")[-1]
    print(f"Env: {save_prefix}, Obs: {obs_shape}, Act: {act_dim}")
    
    config = Config(args)
    
    agent = DreamerV3(obs_shape, act_dim, config)
    agent.world_model.apply(init_weights)
    agent.actor.apply(init_weights)
    agent.critic.apply(init_weights)
    writer = SummaryWriter(log_dir=f"metrics/{save_prefix}")
    
    episode_history = []
    
    avg_reward_window = 50
    score, step = 0, 0
    state, _ = env.reset()
    agent.init_hidden_state()
    
    while len(episode_history) < config.episodes:
        action = agent.act(state)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        agent.store_transition(state, action, reward, done)
        score += reward
        
        step += 1
        
        if done:
            ep = len(episode_history)
            episode_history.append(score)
            if len(agent.replay_buffer) > config.min_buffer_size:
                losses = agent.train()
                writer.add_scalar("Loss/World", losses["world_loss"], ep)
                writer.add_scalar("Loss/Recon", losses["recon_loss"], ep)
                writer.add_scalar("Loss/Reward", losses["reward_loss"], ep)
                writer.add_scalar("Loss/Continue", losses["continue_loss"], ep)
                writer.add_scalar("Loss/KL", losses["kl_loss"], ep)
                writer.add_scalar("Loss/Actor", losses["actor_loss"], ep)
                writer.add_scalar("Loss/Critic", losses["critic_loss"], ep)
                writer.add_scalar("Entropy/Actor", losses["actor_entropy"], ep)
            writer.add_scalar("Reward/Score", score, ep)
            avg_score = np.mean(episode_history[-avg_reward_window:])
            writer.add_scalar("Reward/Average", avg_score, ep)
            
            print(f"[Ep {ep:05d}/{config.episodes}] Score = {score:.2f} Avg.Score = {avg_score:.2f}", end="\r")
            
            if score >= max(episode_history, default=-np.inf):
                agent.save_checkpoint(save_prefix)

            score = 0
            agent.init_hidden_state()
            state, _ = env.reset()
        else:
            state = next_state
    
    print(f"\nFinished training. Final Avg.Score = {avg_score:.2f}")
    writer.close()
    create_animation(args.env, agent)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10000)
    args = parser.parse_args()
    for folder in ["metrics", "environments", "weights", "results"]:
        os.makedirs(folder, exist_ok=True)
    envs = [
        "AirRaidNoFrameskip-v4", "AlienNoFrameskip-v4", "AmidarNoFrameskip-v4",
        "AssaultNoFrameskip-v4", "AsterixNoFrameskip-v4", "AsteroidsNoFrameskip-v4",
        "AtlantisNoFrameskip-v4", "BankHeistNoFrameskip-v4", "BattleZoneNoFrameskip-v4",
        "BeamRiderNoFrameskip-v4", "BerzerkNoFrameskip-v4", "BowlingNoFrameskip-v4",
        "BoxingNoFrameskip-v4", "BreakoutNoFrameskip-v4", "CarnivalNoFrameskip-v4",
        "CentipedeNoFrameskip-v4", "ChopperCommandNoFrameskip-v4", "CrazyClimberNoFrameskip-v4",
        "DefenderNoFrameskip-v4", "DemonAttackNoFrameskip-v4", "DoubleDunkNoFrameskip-v4",
        "ElevatorActionNoFrameskip-v4", "EnduroNoFrameskip-v4", "FishingDerbyNoFrameskip-v4",
        "FreewayNoFrameskip-v4", "FrostbiteNoFrameskip-v4", "GopherNoFrameskip-v4",
        "GravitarNoFrameskip-v4", "HeroNoFrameskip-v4", "IceHockeyNoFrameskip-v4",
        "JamesbondNoFrameskip-v4", "JourneyEscapeNoFrameskip-v4", "KangarooNoFrameskip-v4",
        "KrullNoFrameskip-v4", "KungFuMasterNoFrameskip-v4", "MontezumaRevengeNoFrameskip-v4",
        "MsPacmanNoFrameskip-v4", "NameThisGameNoFrameskip-v4", "PhoenixNoFrameskip-v4",
        "PitfallNoFrameskip-v4", "PongNoFrameskip-v4", "PooyanNoFrameskip-v4",
        "PrivateEyeNoFrameskip-v4", "QbertNoFrameskip-v4", "RiverraidNoFrameskip-v4",
        "RoadRunnerNoFrameskip-v4", "RobotankNoFrameskip-v4", "SeaquestNoFrameskip-v4",
        "SkiingNoFrameskip-v4", "SolarisNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4",
        "StarGunnerNoFrameskip-v4", "TennisNoFrameskip-v4", "TimePilotNoFrameskip-v4",
        "TutankhamNoFrameskip-v4", "UpNDownNoFrameskip-v4", "VentureNoFrameskip-v4",
        "VideoPinballNoFrameskip-v4", "WizardOfWorNoFrameskip-v4", "YarsRevengeNoFrameskip-v4",
        "ZaxxonNoFrameskip-v4", "AdventureNoFrameskip-v4"
    ]
    if args.env:
        train_dreamer(args)
    else:
        rand_order = np.random.permutation(envs)
        for env in rand_order:
            args.env = env
            train_dreamer(args)
