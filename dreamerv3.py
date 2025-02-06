import os
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import imageio
from collections import deque
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.simplefilter("ignore")

def preprocess(image):
    return (image / 255.0).astype(np.float32)

def quantize(image):
    return (image * 255).astype(np.uint8)

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Config:
    def __init__(self, args):
        self.capacity = 100              # number of episodes stored
        self.batch_size = 16             # as per paper (BS = 16)
        self.sequence_length = 64        # (SL = 64)
        self.embed_dim = 1024
        self.latent_dim = 32
        self.num_classes = 32
        self.deter_dim = 512             # GRU units set to 512 per paper
        self.lr = 6e-4
        self.eps = 1e-7
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.discount = 0.99
        self.kl_scale = 0.1
        self.imagination_horizon = 15
        self.min_buffer_size = 10
        self.episodes = args.episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.free_bits = 1.0            # minimum KL per timestep (free bits)
        self.entropy_coef = 0.01        # coefficient for actor entropy bonus

class ReplayBuffer:
    def __init__(self, config, device):
        self.capacity = config.capacity
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.episodes = []
        self.current_episode = []
        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        self.current_episode.append({
            "obs": quantize(obs),
            "action": act,
            "reward": rew,
            "next_obs": next_obs,
            "done": done,
        })
        if done:
            if len(self.current_episode) >= self.sequence_length:
                self.episodes.append(self.current_episode)
            self.current_episode = []
            if len(self.episodes) > self.capacity:
                self.episodes.pop(0)

    def sample(self, n_batches):
        valid_episodes = [ep for ep in self.episodes if len(ep) >= self.sequence_length]
        if not valid_episodes:
            raise StopIteration
        for _ in range(n_batches):
            batch_obs, batch_actions, batch_rewards, batch_dones = [], [], [], []
            for _ in range(self.batch_size):
                ep = valid_episodes[np.random.randint(len(valid_episodes))]
                start_idx = np.random.randint(0, len(ep) - self.sequence_length + 1)
                seq = ep[start_idx : start_idx + self.sequence_length]
                obs_seq = [preprocess(transition["obs"]) for transition in seq]
                action_seq = [transition["action"] for transition in seq]
                reward_seq = [transition["reward"] for transition in seq]
                done_seq = [transition["done"] for transition in seq]
                batch_obs.append(np.array(obs_seq))
                batch_actions.append(np.array(action_seq))
                batch_rewards.append(np.array(reward_seq))
                batch_dones.append(np.array(done_seq))
            yield {
                "observation": torch.tensor(np.array(batch_obs), dtype=torch.float32, device=self.device),
                "action": torch.tensor(np.array(batch_actions), dtype=torch.int64, device=self.device),
                "reward": torch.tensor(np.array(batch_rewards), dtype=torch.float32, device=self.device),
                "done": torch.tensor(np.array(batch_dones), dtype=torch.float32, device=self.device),
            }

    def __len__(self):
        return len(self.episodes)

class ObservationEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2), nn.ReLU(),
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
        self.fc = nn.Linear(feature_dim, 256)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=6, stride=2), nn.Sigmoid(),
        )
        self.output_size = output_size
        self.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 1, 1)
        x = self.deconv(x)
        return F.interpolate(x, size=self.output_size)

class TransitionDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, dist_type="regression"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 200), nn.ReLU(), nn.Linear(200, out_dim)
        )
        self.dist_type = dist_type
        self.apply(init_weights)

    def forward(self, features):
        return self.net(features)

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
            nn.Linear(deter_dim + action_dim, 200),
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
            prior_input = torch.cat([deter, action_seq[t]], dim=-1)
            prior_logits = self.prior_net(prior_input).view(B, self.latent_dim, self.num_classes)
            prior_dist = Categorical(logits=prior_logits)
            post_input = torch.cat([deter, embed_seq[t]], dim=-1)
            posterior_logits = self.posterior_net(post_input).view(B, self.latent_dim, self.num_classes)
            posterior_dist = Categorical(logits=posterior_logits)
            # Straight-through sampling from the categorical
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
            action_dist = actor(feature)
            action = action_dist.sample()
            actions.append(action)
            action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
            stoch_flat = stoch.view(B, -1)
            x = torch.cat([stoch_flat, action_onehot], dim=-1)
            deter = self.gru(x, deter)
            prior_input = torch.cat([deter, action_onehot], dim=-1)
            prior_logits = self.prior_net(prior_input).view(B, self.latent_dim, self.num_classes)
            stoch = F.gumbel_softmax(prior_logits, tau=1.0, hard=True)
        return torch.stack(features, dim=0), torch.stack(actions, dim=0)

class WorldModel(nn.Module):
    def __init__(self, in_channels, action_dim, embed_dim, latent_dim, num_classes, deter_dim, obs_size, lr=1e-4, eps=1e-8):
        super().__init__()
        self.encoder = ObservationEncoder(in_channels, embed_dim)
        self.rssm = RSSM(action_dim, latent_dim, num_classes, deter_dim, embed_dim)
        feature_dim = deter_dim + latent_dim * num_classes
        self.decoder = ObservationDecoder(feature_dim, in_channels, obs_size)
        self.reward_decoder = TransitionDecoder(feature_dim, 1)
        self.continue_decoder = TransitionDecoder(feature_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=eps)

    def observe(self, observations, actions):
        T, B = observations.shape[:2]
        obs_flat = observations.reshape(T * B, *observations.shape[2:])
        embed = self.encoder(obs_flat).view(T, B, -1)
        init_state = self.rssm.init_state(B, observations.device)
        (priors, posteriors), features = self.rssm.observe(embed, actions, init_state)
        feat_dim = features.size(-1)
        features_flat = features.view(T * B, feat_dim)
        recon = self.decoder(features_flat).view(T, B, *observations.shape[2:])
        reward_pred = self.reward_decoder(features_flat)
        continue_pred = self.continue_decoder(features_flat)
        return (priors, posteriors), features, recon, reward_pred, continue_pred

    def imagine(self, init_state, actor, horizon):
        features, actions = self.rssm.imagine(init_state, actor, horizon)
        T, B, feat_dim = features.shape
        features_flat = features.view(T * B, feat_dim)
        reward_pred = self.reward_decoder(features_flat)
        continue_pred = self.continue_decoder(features_flat)
        return features, actions, reward_pred, continue_pred

    def decode(self, features):
        return self.decoder(features)

class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 200), nn.ReLU(), nn.Linear(200, action_dim)
        )
        self.apply(init_weights)

    def forward(self, x):
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)

class Critic(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 200), nn.ReLU(), nn.Linear(200, 1)
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)

class DreamerV3:
    def __init__(self, obs_shape, action_dim, config):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.config = config
        self.replay_buffer = ReplayBuffer(config, config.device)
        self.device = config.device
        in_channels = obs_shape[0]
        self.world_model = WorldModel(
            in_channels, action_dim, config.embed_dim, config.latent_dim,
            config.num_classes, config.deter_dim, obs_shape[1:], lr=config.lr, eps=config.eps
        ).to(self.device)
        feature_dim = config.deter_dim + config.latent_dim * config.num_classes
        self.actor = Actor(feature_dim, action_dim).to(self.device)
        self.critic = Critic(feature_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.hidden_state = None

    def init_hidden_state(self):
        self.hidden_state = self.world_model.rssm.init_state(1, self.device)

    def act(self, observation):
        obs = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.hidden_state is None:
                self.init_hidden_state()
            stoch, deter = self.hidden_state
            feature = torch.cat([deter, stoch.view(1, -1)], dim=-1)
            action_dist = self.actor(feature)
            action = action_dist.sample()
            action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
            x = torch.cat([stoch.view(1, -1), action_onehot], dim=-1)
            deter = self.world_model.rssm.gru(x, deter)
            prior_input = torch.cat([deter, action_onehot], dim=-1)
            prior_logits = self.world_model.rssm.prior_net(prior_input).view(1, self.world_model.rssm.latent_dim, self.world_model.rssm.num_classes)
            stoch = F.gumbel_softmax(prior_logits, tau=1.0, hard=True)
            self.hidden_state = (stoch, deter)
        return int(action.item())

    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay_buffer.store(obs, action, reward, next_obs, done)

    def update_world_model(self, batch):
        obs = batch["observation"].permute(1, 0, 2, 3, 4)
        actions = F.one_hot(batch["action"].long(), num_classes=self.action_dim).float().permute(1, 0, 2)
        rewards = batch["reward"].unsqueeze(-1).permute(1, 0, 2)
        (priors, posteriors), features, recon, reward_pred, terminal_pred = self.world_model.observe(obs, actions)
        recon_loss = F.mse_loss(recon, obs)
        reward_loss = F.mse_loss(reward_pred, rewards.reshape(-1, 1))
        dones = batch["done"].unsqueeze(-1).permute(1, 0, 2)
        terminal_loss = F.binary_cross_entropy_with_logits(terminal_pred, dones.reshape(-1, 1))
        kl_loss = 0
        T = len(priors)
        for t in range(T):
            kl_t = torch.distributions.kl_divergence(posteriors[t], priors[t]).mean()
            # Ensure KL loss is at least free_bits (prevent over-regularization)
            kl_loss += torch.max(kl_t, torch.tensor(self.config.free_bits, device=self.device))
        kl_loss = kl_loss / T
        world_loss = recon_loss + reward_loss + terminal_loss + self.config.kl_scale * kl_loss
        self.world_model.optimizer.zero_grad()
        world_loss.backward()
        self.world_model.optimizer.step()
        return {
            "world_loss": world_loss.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "terminal_loss": terminal_loss.item(),
            "kl_loss": kl_loss.item()
        }

    def update_actor_and_critic(self, init_state):
        horizon = self.config.imagination_horizon
        features, actions, rewards, terminals = self.world_model.imagine(init_state, self.actor, horizon)
        features = features.detach()  # detach world model gradients
        rewards = rewards.detach()
        terminals = terminals.detach()
        T, B, feat_dim = features.shape
        values = self.critic(features.reshape(-1, feat_dim)).reshape(T, B, -1)
        discounts = self.config.discount * (1 - torch.sigmoid(terminals))
        returns = []
        future_return = values[-1]
        for t in reversed(range(T)):
            future_return = rewards[t] + discounts[t] * future_return
            returns.insert(0, future_return)
        returns = torch.stack(returns, dim=0)
        critic_loss = F.mse_loss(symlog(values), symlog(returns.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        features_flat = features.reshape(-1, feat_dim)
        actions_flat = actions.reshape(-1)
        action_dist = self.actor(features_flat)
        log_probs = action_dist.log_prob(actions_flat).reshape(T, B, 1)
        entropy = action_dist.entropy().mean()
        advantages = returns.detach() - values.detach()
        actor_loss = -(log_probs * advantages + self.config.entropy_coef * entropy).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_entropy": entropy.item()
        }

    def train(self):
        losses = {"world_loss": 0, "recon_loss": 0, "reward_loss": 0,
                  "terminal_loss": 0, "kl_loss": 0,
                  "actor_loss": 0, "critic_loss": 0, "actor_entropy": 0}

        batch = next(iter(self.replay_buffer.sample(1)))

        wm_losses = self.update_world_model(batch)
        for key in ["world_loss", "recon_loss", "reward_loss", "terminal_loss", "kl_loss"]:
            losses[key] += wm_losses[key]
        B = batch["observation"].shape[0]
        init_state = self.world_model.rssm.init_state(B, self.device)
        ac_losses = self.update_actor_and_critic(init_state)
        for key in ["actor_loss", "critic_loss", "actor_entropy"]:
            losses[key] += ac_losses[key]

        return losses
    
    def save_checkpoint(self, env_name):
        torch.save(self.world_model.state_dict(), f"weights/{env_name}_world_model.pt")
        torch.save(self.actor.state_dict(), f"weights/{env_name}_actor.pt")
        torch.save(self.critic.state_dict(), f"weights/{env_name}_critic.pt")

    def load_checkpoint(self, env_name):
        self.world_model.load_state_dict(
            torch.load(f"weights/{env_name}_world_model.pt")
        )
        self.actor.load_state_dict(torch.load(f"weights/{env_name}_actor.pt"))
        self.critic.load_state_dict(torch.load(f"weights/{env_name}_critic.pt"))

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
    writer = SummaryWriter(log_dir=f"metrics/{save_prefix}")
    
    episode_history = []
    
    avg_reward_window = 100
    score, step = 0, 0
    state, _ = env.reset()
    agent.init_hidden_state()
    
    while len(episode_history) < config.episodes:
        action = agent.act(state)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        agent.store_transition(state, action, reward, next_state, done)
        score += reward
        
        if len(agent.replay_buffer) > config.min_buffer_size and step % 10 == 0:
            losses = agent.train()
            writer.add_scalar("Loss/World", losses["world_loss"], step)
            writer.add_scalar("Loss/Recon", losses["recon_loss"], step)
            writer.add_scalar("Loss/Reward", losses["reward_loss"], step)
            writer.add_scalar("Loss/Terminal", losses["terminal_loss"], step)
            writer.add_scalar("Loss/KL", losses["kl_loss"], step)
            writer.add_scalar("Loss/Actor", losses["actor_loss"], step)
            writer.add_scalar("Loss/Critic", losses["critic_loss"], step)
            writer.add_scalar("Entropy/Actor", losses["actor_entropy"], step)
        step += 1
        
        if done:
            episode_history.append(score)
            writer.add_scalar("Reward/Score", score, len(episode_history))
            score = 0
            agent.init_hidden_state()
            state, _ = env.reset()
        else:
            state = next_state

        if episode_history:
            avg_score = np.mean(episode_history[-avg_reward_window:])
            writer.add_scalar("Reward/Average", avg_score, len(episode_history))
            print(f"[Ep {len(episode_history):05d}/{config.episodes}] Avg.Score = {avg_score:.2f}", end="\r")
            
            if avg_score >= max(episode_history, default=-np.inf):
                agent.save_checkpoint(save_prefix)
    
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
