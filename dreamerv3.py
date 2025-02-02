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
from torch.distributions import Normal, Categorical, Bernoulli


def preprocess(image):
    # Normalize pixel values from [0, 255] to [-0.5, 0.5]
    return image / 255.0 - 0.5


def quantize(image):
    # Inverse of preprocess: map [-0.5, 0.5] back to [0, 255] as uint8.
    return ((image + 0.5) * 255).astype(np.uint8)


class Config:
    def __init__(self, args):
        self.capacity = 1000
        self.batch_size = 32
        self.sequence_length = 50
        self.embed_dim = 1024
        self.latent_dim = 30
        self.deter_dim = 200
        self.lr = 6e-4
        self.eps = 1e-7
        self.actor_lr = 1e-4
        self.critic_lr = 1e-4
        self.discount = 0.99
        self.kl_scale = 1.0
        self.imagination_horizon = 15
        self.train_horizon = args.train_horizon
        self.num_updates = args.num_updates
        self.min_buffer_size = 1000
        self.n_envs = 1
        self.episodes = args.episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, config, device):
        self.capacity = config.capacity
        self.batch_size = config.batch_size
        self.seq_length = config.sequence_length
        self.device = device
        self.episodes = deque(maxlen=self.capacity)
        self._ep = {"observation": [], "action": [], "reward": [], "done": []}

    def store(self, obs, act, rew, next_obs, done):
        self._ep["observation"].append(obs)
        self._ep["action"].append(act)
        self._ep["reward"].append(rew)
        self._ep["done"].append(done)
        if done:
            # Bootstrap next observation
            self._ep["observation"].append(next_obs)
            ep = {k: np.array(v) for k, v in self._ep.items()}
            ep["observation"] = quantize(ep["observation"])
            self.episodes.append(ep)
            self._ep = {"observation": [], "action": [], "reward": [], "done": []}

    def sample(self, n_batches):
        for _ in range(n_batches):
            batch = {k: [] for k in ["observation", "action", "reward", "done"]}
            for _ in range(self.batch_size):
                ep = np.random.choice(self.episodes)
                t = np.random.randint(len(ep["observation"]) - self.seq_length)
                for k in batch.keys():
                    batch[k].append(ep[k][t : t + self.seq_length])
            for k in batch.keys():
                batch[k] = torch.tensor(
                    np.array(batch[k]), device=self.device, dtype=torch.float32
                )
            batch["observation"] = preprocess(batch["observation"])
            yield batch

    def __len__(self):
        return len(self.episodes)


class ObservationEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ObservationDecoder(nn.Module):
    def __init__(self, feature_dim, out_channels, output_size):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 256)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        self.output_size = output_size

    def forward(self, x):
        # x: (B, feature_dim)
        x = self.fc(x)
        x = x.view(x.size(0), 256, 1, 1)
        x = self.deconv(x)
        x = nn.functional.interpolate(x, size=self.output_size)
        return x


class TransitionDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, dist_type="normal"):
        super().__init__()
        self.dist_type = dist_type
        self.net = nn.Sequential(
            nn.Linear(in_dim, 200), nn.ReLU(), nn.Linear(200, out_dim)
        )
        if dist_type == "normal":
            self.std = nn.Parameter(torch.zeros(out_dim))

    def forward(self, features):
        x = self.net(features)
        if self.dist_type == "normal":
            std = torch.exp(self.std)
            return Normal(x, std)
        elif self.dist_type == "bernoulli":
            return Bernoulli(logits=x)


class Prior(nn.Module):
    def __init__(self, action_dim, latent_dim, deter_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deter_dim + action_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 2 * latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, deter, action):
        x = torch.cat([deter, action], dim=-1)
        stats = self.net(x)
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        std = torch.exp(log_std.clamp(-4, 15))
        return Normal(mean, std)


class Posterior(nn.Module):
    def __init__(self, deter_dim, embed_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 2 * latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, deter, embed):
        x = torch.cat([deter, embed], dim=-1)
        stats = self.net(x)
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        std = torch.exp(log_std.clamp(-4, 15))
        return Normal(mean, std)


class RSSM(nn.Module):
    def __init__(self, action_dim, latent_dim, deter_dim, embed_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.deter_dim = deter_dim
        self.embed_dim = embed_dim
        self.gru = nn.GRUCell(latent_dim + action_dim, deter_dim)
        self.prior = Prior(action_dim, latent_dim, deter_dim)
        self.posterior = Posterior(deter_dim, embed_dim, latent_dim)

    def init_state(self, batch_size, device):
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.latent_dim, device=device)
        return (stoch, deter)

    def observe(self, embed_seq, action_seq, init_state):
        # embed_seq: (T, B, embed_dim); action_seq: (T, B, action_dim)
        T, B = action_seq.shape[:2]
        priors, posteriors, features = [], [], []
        stoch, deter = init_state
        for t in range(T):
            x = torch.cat([stoch, action_seq[t]], dim=-1)
            deter = self.gru(x, deter)
            prior_dist = self.prior(deter, action_seq[t])
            stoch_prior = prior_dist.rsample()
            post_dist = self.posterior(deter, embed_seq[t])
            stoch = post_dist.rsample()
            features.append(torch.cat([deter, stoch], dim=-1))
            priors.append(prior_dist)
            posteriors.append(post_dist)
        features = torch.stack(features, dim=0)
        return (priors, posteriors), features

    def imagine(self, init_state, actor, horizon):
        # Generate imagined latent rollouts given an actor
        stoch, deter = init_state
        features, actions = [], []
        for _ in range(horizon):
            feature = torch.cat([deter, stoch], dim=-1)
            features.append(feature)
            action_dist = actor(feature)
            action = action_dist.sample()
            actions.append(action)
            action_onehot = F.one_hot(
                action, num_classes=actor.net[-1].out_features
            ).float()
            x = torch.cat([stoch, action_onehot], dim=-1)
            deter = self.gru(x, deter)
            prior_dist = self.prior(deter, action_onehot)
            stoch = prior_dist.rsample()
        features = torch.stack(features, dim=0)
        actions = torch.stack(actions, dim=0)
        return features, actions


class WorldModel(nn.Module):
    def __init__(
        self,
        in_channels,
        action_dim,
        embed_dim,
        latent_dim,
        deter_dim,
        obs_size,
        lr=6e-4,
        eps=1e-7,
    ):
        super().__init__()
        self.encoder = ObservationEncoder(in_channels, embed_dim)
        # Decoder reconstructs observation from latent features (deter + stoch)
        self.decoder = ObservationDecoder(deter_dim + latent_dim, in_channels, obs_size)
        self.reward_decoder = TransitionDecoder(deter_dim + latent_dim, 1, "normal")
        self.terminal_decoder = TransitionDecoder(
            deter_dim + latent_dim, 1, "bernoulli"
        )
        self.rssm = RSSM(action_dim, latent_dim, deter_dim, embed_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=eps)

    def observe(self, observations, actions):
        # observations: (T, B, C, H, W), actions: (T, B, action_dim)
        T, B = observations.shape[:2]
        obs_flat = observations.view(T * B, *observations.shape[2:])
        embed_flat = self.encoder(obs_flat)
        embed = embed_flat.view(T, B, -1)
        init_state = self.rssm.init_state(B, observations.device)
        (priors, posteriors), features = self.rssm.observe(embed, actions, init_state)
        feat_dim = features.shape[-1]
        features_flat = features.view(T * B, feat_dim)
        recon_flat = self.decoder(features_flat)
        recon = recon_flat.view(T, B, *observations.shape[2:])
        reward_dist = self.reward_decoder(features_flat)
        reward = reward_dist.mean.view(T, B, -1)
        terminal_dist = self.terminal_decoder(features_flat)
        terminal = terminal_dist.logits.view(T, B, -1)
        return (priors, posteriors), features, recon, reward, terminal

    def imagine(self, init_state, actor, horizon):
        features, actions = self.rssm.imagine(init_state, actor, horizon)
        T, B, feat_dim = features.shape
        features_flat = features.view(T * B, feat_dim)
        reward_dist = self.reward_decoder(features_flat)
        reward = reward_dist.mean.view(T, B, -1)
        terminal_dist = self.terminal_decoder(features_flat)
        terminal = terminal_dist.logits.view(T, B, -1)
        return features, actions, reward, terminal

    def decode(self, features):
        return self.decoder(features)


class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 200), nn.ReLU(), nn.Linear(200, action_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        return Categorical(logits=logits)


class Critic(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 200), nn.ReLU(), nn.Linear(200, 1)
        )

    def forward(self, x):
        return self.net(x)


class DreamerV3:
    def __init__(self, obs_shape, action_dim, config):
        self.obs_shape = obs_shape  # (C, H, W)
        self.action_dim = action_dim
        self.config = config
        self.device = config.device
        in_channels = obs_shape[0]
        self.world_model = WorldModel(
            in_channels,
            action_dim,
            embed_dim=config.embed_dim,
            latent_dim=config.latent_dim,
            deter_dim=config.deter_dim,
            obs_size=obs_shape[1:],
            lr=config.lr,
            eps=config.eps,
        ).to(self.device)
        feat_dim = config.deter_dim + config.latent_dim
        self.actor = Actor(feat_dim, action_dim).to(self.device)
        self.critic = Critic(feat_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )
        self.replay_buffer = ReplayBuffer(config, self.device)
        self.hidden_state = None  # (stoch, deter)

    def init_hidden_state(self):
        self.hidden_state = self.world_model.rssm.init_state(1, self.device)

    def act(self, observation):
        # observation: (C, H, W); add batch dim
        obs = torch.tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            _ = self.world_model.encoder(obs)
            if self.hidden_state is None:
                self.init_hidden_state()
            stoch, deter = self.hidden_state
            feature = torch.cat([deter, stoch], dim=-1)
            action_dist = self.actor(feature)
            action = action_dist.sample()
            # Convert discrete action into one-hot for dynamics
            action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
            x = torch.cat([stoch, action_onehot], dim=-1)
            deter = self.world_model.rssm.gru(x, deter)
            prior_dist = self.world_model.rssm.prior(deter, action_onehot)
            stoch = prior_dist.rsample()
            self.hidden_state = (stoch, deter)
        return int(action.item())

    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay_buffer.store(obs, action, reward, next_obs, done)

    def update_world_model(self, batch):
        # Rearranging batch: (B, T, ...)
        obs = batch["observation"].permute(1, 0, 2, 3, 4)
        # actions: convert to one-hot
        actions = F.one_hot(batch["action"].long(), num_classes=self.action_dim).float()
        actions = actions.permute(1, 0, 2)
        rewards = batch["reward"].unsqueeze(-1).permute(1, 0, 2)
        (priors, posteriors), features, recon, reward_pred, terminal_pred = (
            self.world_model.observe(obs, actions)
        )
        recon_loss = F.mse_loss(recon, obs)
        reward_loss = F.mse_loss(reward_pred, rewards)
        dones = batch["done"].unsqueeze(-1).permute(1, 0, 2)
        terminal_loss = F.binary_cross_entropy_with_logits(terminal_pred, dones)
        kl_loss = 0
        T = len(priors)
        for t in range(T):
            kl_loss += torch.distributions.kl_divergence(
                posteriors[t], priors[t]
            ).mean()
        kl_loss = kl_loss / T
        world_loss = (
            recon_loss + reward_loss + terminal_loss + self.config.kl_scale * kl_loss
        )
        self.world_model.optimizer.zero_grad()
        world_loss.backward()
        self.world_model.optimizer.step()
        return {"world_loss": world_loss.item()}

    def update_actor_and_critic(self, init_state):
        horizon = self.config.imagination_horizon
        features, actions, rewards, terminals = self.world_model.imagine(
            init_state, self.actor, horizon
        )
        T, B, feat_dim = features.shape
        values = self.critic(features.view(-1, feat_dim)).view(T, B, -1)
        # Use sigmoid to obtain termination probability and compute effective discount
        discounts = self.config.discount * (1 - torch.sigmoid(terminals))
        returns = []
        future_return = values[-1]
        for t in reversed(range(T)):
            future_return = rewards[t] + discounts[t] * future_return
            returns.insert(0, future_return)
        returns = torch.stack(returns, dim=0)
        critic_loss = F.mse_loss(values, returns.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -(returns - values.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def train(self, num_updates=1):
        losses = {"world_loss": 0, "actor_loss": 0, "critic_loss": 0}
        for _ in range(num_updates):
            try:
                batch = next(iter(self.replay_buffer.sample(1)))
            except StopIteration:
                continue
            wm_losses = self.update_world_model(batch)
            losses["world_loss"] += wm_losses["world_loss"]
            B = batch["observation"].shape[0]
            init_state = self.world_model.rssm.init_state(B, self.device)
            ac_losses = self.update_actor_and_critic(init_state)
            losses["actor_loss"] += ac_losses["actor_loss"]
            losses["critic_loss"] += ac_losses["critic_loss"]
        losses = {k: v / num_updates for k, v in losses.items()}
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
        self.frame_buffer = np.zeros(
            (2, *env.observation_space.shape), dtype=np.float32
        )

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
        self.frame_buffer = np.zeros(
            (2, *self.env.observation_space.shape), dtype=np.float32
        )
        self.frame_buffer[0] = state
        return state, info


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(0.0, 1.0, self.shape, dtype=np.float32)

    def observation(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, self.shape, interpolation=cv2.INTER_AREA)
        return state / 255.0


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, size=4):
        super().__init__(env)
        self.size = int(size)
        self.stack = deque([], maxlen=self.size)
        shape = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (self.size, *shape), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        self.stack = deque([state] * self.size, maxlen=self.size)
        return np.array(self.stack), info

    def observation(self, state):
        self.stack.append(state)
        return np.array(self.stack)


class AtariEnv:
    def __init__(
        self,
        env_id,
        shape=(64, 64),
        repeat=4,
        clip_rewards=False,
        no_ops=0,
        fire_first=False,
    ):
        base_env = gym.make(env_id, render_mode="rgb_array")
        env = RepeatActionAndMaxFrame(
            base_env, repeat, clip_rewards, no_ops, fire_first
        )
        env = PreprocessFrame(env, shape)
        env = StackFrames(env, repeat)
        self.env = env

    def make(self):
        return self.env


def plot_results(rewards, world_losses, actor_losses, critic_losses, save_prefix):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.plot(rewards, label="Episode Reward")
    if len(rewards) >= 100:
        running_avg = np.convolve(rewards, np.ones(100) / 100, mode="valid")
        ax1.plot(running_avg, label="Running Average (100 episodes)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Rewards over Episodes")
    ax1.legend()
    ax2.plot(world_losses, label="World Model Loss")
    ax2.plot(actor_losses, label="Actor Loss")
    ax2.plot(critic_losses, label="Critic Loss")
    ax2.set_xlabel("Update Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("Losses over Update Steps")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"results/{save_prefix}.png")
    plt.close()


def save_animation(frames, filename):
    with imageio.get_writer(filename, mode="I", loop=0) as writer:
        for frame in frames:
            writer.append_data(frame)


def create_animation(env_name, agent, seeds=100):
    agent.load_checkpoint(env_name)
    env = AtariEnv(env_name, shape=(42, 42), repeat=4, clip_rewards=False).make()
    save_prefix = env_name.split("/")[-1]
    best_total_reward, best_frames = float("-inf"), None
    for s in range(seeds):
        state, _ = env.reset()
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
    env_instance = AtariEnv(args.env, shape=(64, 64), repeat=4, clip_rewards=False)
    env = env_instance.make()
    obs_shape = env.observation_space.shape  # (stack, H, W)
    act_dim = env.action_space.n
    save_prefix = args.env.split("/")[-1]
    print(f"\nEnvironment: {save_prefix}")
    print(f"Obs.Space: {obs_shape}")
    print(f"Act.Space: {act_dim}")
    config = Config(args)
    agent = DreamerV3(obs_shape, act_dim, config)
    world_losses, actor_losses, critic_losses = [], [], []
    best_avg_reward, frame_idx = float("-inf"), 0
    avg_reward_window = 100
    history = []
    score = 0
    state, _ = env.reset()
    agent.init_hidden_state()

    while len(history) < config.episodes:
        action = agent.act(state)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        agent.store_transition(state, action, reward, next_state, done)
        score += reward
        if done:
            history.append(score)
            score = 0
            agent.init_hidden_state()
            state, _ = env.reset()
        else:
            state = next_state

        if (
            len(agent.replay_buffer) > config.min_buffer_size
            and frame_idx % config.train_horizon == 0
        ):
            losses = agent.train(num_updates=config.num_updates)
            if losses:
                world_losses.append(losses["world_loss"])
                actor_losses.append(losses["actor_loss"])
                critic_losses.append(losses["critic_loss"])
        frame_idx += 1
        if history:
            avg_score = np.mean(history[-avg_reward_window:])
            print(
                f"[Episode {len(history):05d}/{config.episodes}]  Avg.Score = {avg_score:.2f}",
                end="\r",
            )
            if avg_score > best_avg_reward:
                best_avg_reward = avg_score
                agent.save_checkpoint(save_prefix)

    torch.save(
        agent.world_model.state_dict(), f"weights/{save_prefix}_world_model_final.pth"
    )
    torch.save(agent.actor.state_dict(), f"weights/{save_prefix}_actor_final.pth")
    torch.save(agent.critic.state_dict(), f"weights/{save_prefix}_critic_final.pth")
    plot_results(history, world_losses, actor_losses, critic_losses, save_prefix)
    create_animation(args.env, agent)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default=None, help="Environment ID (e.g. ALE/Breakout-v5)"
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num_updates", type=int, default=1)
    parser.add_argument("--train_horizon", type=int, default=50)
    args = parser.parse_args()

    for folder in ["metrics", "environments", "weights", "results"]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    default_envs = [
        "ALE/Breakout-v5",
        "ALE/Pong-v5",
        "ALE/SpaceInvaders-v5",
        "ALE/Enduro-v5",
    ]
    if args.env:
        train_dreamer(args)
    else:
        for env_name in default_envs:
            args.env = env_name
            train_dreamer(args)
