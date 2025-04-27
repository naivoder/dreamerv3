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
from datetime import datetime
from collections import deque
from torch.distributions import Categorical, Independent
from torch.utils.tensorboard import SummaryWriter
import warnings
from torch.cuda.amp import autocast, GradScaler

warnings.simplefilter("ignore")
gym.register_envs(ale_py)
torch.backends.cudnn.benchmark = True

def preprocess(image):
    return image.astype(np.float32) / 255.0

def quantize(image):
    return (image * 255).clip(0, 255).astype(np.uint8)

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1e-5)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.clamp(torch.abs(x), max=20.0)) - 1)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Config:
    def __init__(self, args):
        self.capacity = 1_000_000
        self.batch_size = 2 # 16
        self.sequence_length = 32 # 64 on machine with more RAM...
        self.embed_dim = 1024
        self.latent_dim = 32
        self.num_classes = 32
        self.deter_dim = 512
        self.lr = 3e-4
        self.eps = 1e-5
        self.actor_lr = 3e-5
        self.critic_lr = 3e-5
        self.discount = 0.99
        self.kl_scale = 0.1 # 1.0
        self.imagination_horizon = 15
        self.min_buffer_size = 5000
        self.episodes = args.episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.free_bits = 1.0
        self.entropy_coef = 0.001 # 0.01 
        self.updates_per_step = 10
        self.grad_clip = 100.0
        self.mixed_precision = True

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
        return self.probs + (samples - self.probs).detach()

class ObservationEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, embed_dim),  # Corrected input dimensions
            nn.LayerNorm(embed_dim)
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.conv(x)

class ObservationDecoder(nn.Module):
    def __init__(self, feature_dim, out_channels, output_size):
        super().__init__()
        self.out_channels = out_channels
        self.output_size = output_size
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256*8*8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels * 255, kernel_size=3, padding=1)
        )
        self.apply(init_weights)

    def forward(self, x):
        x = self.net(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.out_channels, 255, *self.output_size)
        x = x.permute(0, 1, 3, 4, 2) 
        return Independent(
            OneHotCategoricalStraightThrough(logits=x),
            reinterpreted_batch_ndims=3
        )

class TwoHotCategoricalStraightThrough(torch.distributions.Distribution):
    def __init__(self, logits, bins=255, low=-20.0, high=20.0):
        super().__init__(validate_args=False)
        self.logits = logits
        self.bin_centers = torch.linspace(low, high, bins, device=logits.device)

    def log_prob(self, value):
        value = symlog(value).clamp(self.bin_centers[0], self.bin_centers[-1])
        indices = ((value - self.bin_centers[0]) / 
                 (self.bin_centers[1] - self.bin_centers[0])).clamp(0, len(self.bin_centers)-1)
        
        lower = indices.floor().long().unsqueeze(-1)
        upper = indices.ceil().long().unsqueeze(-1)
        alpha = (indices - lower.squeeze(-1)).unsqueeze(-1)

        probs = F.softmax(self.logits, dim=-1)
        return torch.log(
            (1 - alpha) * probs.gather(-1, lower) + 
            alpha * probs.gather(-1, upper)
        ).squeeze(-1)

    @property
    def mean(self):
        return symexp((F.softmax(self.logits, dim=-1) * self.bin_centers).sum(-1, keepdim=True))

class RSSM(nn.Module):
    def __init__(self, action_dim, latent_dim, num_classes, deter_dim, embed_dim):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.deter_dim = deter_dim
        self.embed_dim = embed_dim
        
        # Separate networks for prior and posterior
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, 200),
            nn.LayerNorm(200),
            nn.ReLU(),
            nn.Linear(200, latent_dim * num_classes)
        )
        
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, 200),
            nn.LayerNorm(200),
            nn.ReLU(),
            nn.Linear(200, latent_dim * num_classes)
        )
        
        self.gru = nn.GRUCell(latent_dim * num_classes + action_dim, deter_dim)
        self.deter_init = nn.Parameter(torch.zeros(1, deter_dim))
        self.stoch_init = nn.Parameter(torch.zeros(1, latent_dim * num_classes))
        self.apply(init_weights)

    def init_state(self, batch_size, device):
        stoch = F.one_hot(torch.zeros(batch_size, self.latent_dim, dtype=torch.long), 
                         self.num_classes).float().to(device)
        deter = self.deter_init.repeat(batch_size, 1)
        return (stoch, deter)

    def imagine_step(self, stoch, deter, action):
        action_oh = F.one_hot(action, self.action_dim).float()
        gru_input = torch.cat([stoch.flatten(1), action_oh], dim=1)
        deter = self.gru(gru_input, deter)
        
        # Use prior network
        prior_logits = self.prior_net(deter).view(-1, self.latent_dim, self.num_classes)
        stoch = F.gumbel_softmax(prior_logits, tau=1.0, hard=True)
        return stoch, deter

    def observe_step(self, deter, embed):
        # Use posterior network
        post_logits = self.post_net(torch.cat([deter, embed], dim=1))
        post_logits = post_logits.view(-1, self.latent_dim, self.num_classes)
        return post_logits

    def imagine(self, init_state, actor, horizon):
        stoch, deter = init_state
        features, actions = [], []
        
        for _ in range(horizon):
            feature = torch.cat([deter, stoch.flatten(1)], dim=1)
            with torch.no_grad():
                action = actor(feature).sample()
            
            stoch, deter = self.imagine_step(stoch, deter, action)
            features.append(feature)
            actions.append(action)
        
        return torch.stack(features), torch.stack(actions)

    def observe(self, embed_seq, action_seq, init_state):
        priors, posteriors = [], []
        features = []
        stoch, deter = init_state
        
        for t in range(action_seq.size(0)):
            # GRU update
            gru_input = torch.cat([stoch.flatten(1), action_seq[t]], dim=1)
            deter = self.gru(gru_input, deter)
            
            # Get prior/posterior
            prior_logits, post_logits = self.observe_step(deter, embed_seq[t])
            
            prior_dist = Independent(OneHotCategoricalStraightThrough(logits=prior_logits), 1)
            post_dist = Independent(OneHotCategoricalStraightThrough(logits=post_logits), 1)
            
            # Sample posterior
            stoch = F.gumbel_softmax(post_logits, tau=1.0, hard=True)
            features.append(torch.cat([deter, stoch.flatten(1)], dim=1))
            
            priors.append(prior_dist)
            posteriors.append(post_dist)
        
        return (priors, posteriors), torch.stack(features)

class WorldModel(nn.Module):
    def __init__(self, in_channels, action_dim, embed_dim, latent_dim, num_classes, deter_dim, obs_size):
        super().__init__()
        self.encoder = ObservationEncoder(in_channels, embed_dim)
        self.rssm = RSSM(action_dim, latent_dim, num_classes, deter_dim, embed_dim)
        self.decoder = ObservationDecoder(deter_dim + latent_dim * num_classes, in_channels, obs_size[1:])
        self.reward_decoder = nn.Sequential(nn.Linear(deter_dim + latent_dim * num_classes, 255))
        self.continue_decoder = nn.Sequential(nn.Linear(deter_dim + latent_dim * num_classes, 1), nn.Sigmoid())

    def observe(self, observations, actions):
        with torch.amp.autocast("cuda"):
            embed = self.encoder(observations.flatten(0, 1)).view(actions.size(0), actions.size(1), -1)
            actions_onehot = F.one_hot(actions, self.rssm.action_dim).float()
            
            priors, posteriors = [], []
            features = []
            stoch, deter = self.rssm.init_state(actions.size(1), observations.device)
            
            for t in range(actions.size(0)):
                deter = self.rssm.gru(torch.cat([stoch.flatten(1), actions_onehot[t]], dim=1), deter)
                
                # Get prior from prior network
                prior_logits = self.rssm.prior_net(deter).view(deter.size(0), 
                                                             self.rssm.latent_dim, 
                                                             self.rssm.num_classes)
                prior_dist = Independent(OneHotCategoricalStraightThrough(logits=prior_logits), 1)
                
                # Get posterior from posterior network
                post_logits = self.rssm.post_net(torch.cat([deter, embed[t]], dim=1))
                post_logits = post_logits.view(deter.size(0), 
                                             self.rssm.latent_dim, 
                                             self.rssm.num_classes)
                post_dist = Independent(OneHotCategoricalStraightThrough(logits=post_logits), 1)
                
                stoch = F.gumbel_softmax(post_logits, tau=1.0, hard=True)
                features.append(torch.cat([deter, stoch.flatten(1)], dim=1))
                priors.append(prior_dist)
                posteriors.append(post_dist)
            
            features = torch.stack(features)
            recon_dist = self.decoder(features.flatten(0, 1))
            reward_dist = TwoHotCategoricalStraightThrough(self.reward_decoder(features.flatten(0, 1)))
            continue_pred = self.continue_decoder(features.flatten(0, 1))
            
        return (priors, posteriors), features, recon_dist, reward_dist, continue_pred

class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 200), nn.LayerNorm(200), nn.ReLU(),
            nn.Linear(200, 200), nn.LayerNorm(200), nn.ReLU(),
            nn.Linear(200, action_dim))
        self.apply(init_weights)

    def forward(self, x):
        return Categorical(logits=self.net(x))

class Critic(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 200), nn.LayerNorm(200), nn.ReLU(),
            nn.Linear(200, 200), nn.LayerNorm(200), nn.ReLU(),
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
            config.num_classes, config.deter_dim, obs_shape
        ).to(self.device)
        
        feature_dim = config.deter_dim + config.latent_dim * config.num_classes
        self.actor = Actor(feature_dim, action_dim).to(self.device)
        self.critic = Critic(feature_dim).to(self.device)
        
        self.optimizers = {
            'world': optim.Adam(self.world_model.parameters(), lr=config.lr, eps=config.eps),
            'actor': optim.Adam(self.actor.parameters(), lr=config.actor_lr),
            'critic': optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        }
        self.scalers = {k: torch.amp.GradScaler("CUDA") for k in self.optimizers}
        
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
            
            # Get posterior
            post_logits = self.world_model.rssm.observe_step(deter, embed)
            post_logits = post_logits.view(1, self.config.latent_dim, self.config.num_classes)
            stoch = F.gumbel_softmax(post_logits, tau=1.0, hard=True)
            
            # Get action
            feature = torch.cat([deter, stoch.flatten(1)], dim=1)
            action = self.actor(feature).sample()
            
            # Update hidden state
            _, deter = self.world_model.rssm.imagine_step(stoch, deter, action)
            self.hidden_state = (stoch, deter)
        
        return int(action.item())

    def store_transition(self, obs, action, reward, done):
        self.replay_buffer.store(quantize(obs), action, reward, done)

    def update_world_model(self, batch):
        self.optimizers['world'].zero_grad()
        
        with torch.amp.autocast("cuda"):
            (priors, posteriors), features, recon_dist, reward_dist, continue_pred = self.world_model.observe(
                batch['observation'], batch['action'])
            
            # Calculate losses
            recon_loss = -recon_dist.log_prob((batch['observation'] * 255).long().flatten(0, 1)).mean()
            reward_loss = -reward_dist.log_prob(batch['reward'].flatten(0, 1)).mean()
            continue_loss = F.binary_cross_entropy_with_logits(continue_pred.flatten(0, 1), (1 - batch['done'].flatten(0, 1)))
            
            # KL loss with free bits
            kl_loss = 0
            for prior, posterior in zip(priors, posteriors):
                kl_t = torch.distributions.kl_divergence(posterior, prior)
                kl_t = torch.mean(kl_t, dim=0)  # Average over batch, keep latents
                kl_t = torch.sum(torch.clamp(kl_t, min=self.config.free_bits))
                kl_loss += kl_t
            kl_loss /= len(priors)
            
            total_loss = recon_loss + reward_loss + continue_loss + self.config.kl_scale * kl_loss
        
        self.scalers['world'].scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.grad_clip)
        self.scalers['world'].step(self.optimizers['world'])
        self.scalers['world'].update()
        
        return {
            "world_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item(),
            "kl_loss": kl_loss.item()
        }

    def update_actor_and_critic(self):
        B = self.config.batch_size
        init_state = self.world_model.rssm.init_state(B, self.device)
        
        with torch.no_grad(), torch.amp.autocast("cuda"):
            features, actions = self.world_model.rssm.imagine(init_state, self.actor, self.config.imagination_horizon)
            
            # Predict rewards and continues
            reward_dist = TwoHotCategoricalStraightThrough(self.world_model.reward_decoder(features.flatten(0, 1)))
            continue_pred = self.world_model.continue_decoder(features.flatten(0, 1))
            
            rewards = reward_dist.mean.view_as(actions)
            continues = continue_pred.view_as(actions)
            discounts = self.config.discount * continues
        
        # Calculate lambda returns
        T, B, feat_dim = features.shape
        features_flat = features.reshape(-1, feat_dim)
        
        # Critic update
        self.optimizers['critic'].zero_grad()
        with torch.amp.autocast("cuda"):
            values = self.critic(features_flat).reshape(T, B)
            
            returns = torch.zeros_like(values)
            last = values[-1]
            for t in reversed(range(T)):
                last = returns[t] = rewards[t] + discounts[t] * last
            
            critic_loss = F.mse_loss(values, returns.detach())
        
        self.scalers['critic'].scale(critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.scalers['critic'].step(self.optimizers['critic'])
        self.scalers['critic'].update()
        
        # Actor update
        self.optimizers['actor'].zero_grad()
        with torch.amp.autocast("cuda"):
            advantages = returns - values
            action_dist = self.actor(features_flat)
            log_probs = action_dist.log_prob(actions.reshape(-1))
            entropy = action_dist.entropy().mean()
            
            actor_loss = -(log_probs * advantages.reshape(-1).detach()).mean() - self.config.entropy_coef * entropy
        
        self.scalers['actor'].scale(actor_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.scalers['actor'].step(self.optimizers['actor'])
        self.scalers['actor'].update()
        
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
            ac_losses = self.update_actor_and_critic()
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
    env = AtariEnv(env_name, shape=(64, 64), repeat=4, clip_rewards=False).make()
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"metrics/{save_prefix}_{timestamp}")

    for key, value in vars(config).items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f'config/{key}', value, 0)  
        else:
            writer.add_text(f'config/{key}', str(value), 0)
    
    episode_history = []
    
    avg_reward_window = 50
    score, step = 0, 0
    best_avg = float("-inf")
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
                agent.save_checkpoint(save_prefix+"_best")

            if avg_score >= best_avg:
                best_avg = avg_score
                agent.save_checkpoint(save_prefix+"_best_avg")

            score = 0
            agent.init_hidden_state()
            state, _ = env.reset()
        else:
            state = next_state
    
    print(f"\nFinished training. Final Avg.Score = {avg_score:.2f}")
    agent.save_checkpoint(save_prefix+"_final")
    writer.close()
    env.close()
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