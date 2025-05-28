import os
import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.distributions import Categorical, Independent
from environment import ENV_LIST
import utils
import warnings

warnings.simplefilter("ignore")
gym.register_envs(ale_py)
torch.backends.cudnn.benchmark = True


class Config:
    def __init__(self, args):
        self.capacity = 2_000_000
        self.batch_size = 16
        self.sequence_length = 64
        self.embed_dim = 1024
        self.latent_dim = 32
        self.num_classes = 32
        self.deter_dim = 4096
        self.lr = 4e-5
        self.eps = 1e-20
        self.actor_lr = 4e-5
        self.critic_lr = 4e-5
        self.discount = 0.997
        self.gae_lambda = 0.95
        self.rep_loss_scale = 0.1
        self.imagination_horizon = 15
        self.min_buffer_size = 500
        self.episodes = 100_000
        self.device = torch.device("cuda")
        self.free_bits = 1.0
        self.entropy_coef = 3e-4
        self.retnorm_scale = 1.0
        self.retnorm_limit = 1.0
        self.retnorm_decay = 0.99
        self.critic_ema_decay = 0.98
        self.update_interval = 2
        self.updates_per_step = 1
        self.mixed_precision = True
        self.wandb_key = args.wandb_key


class ReplayBuffer:
    def __init__(self, config, device, obs_shape):
        self.num_envs = 16
        self.capacity = config.capacity // self.num_envs
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.device = device
        self.obs_shape = obs_shape

        self.obs_buf = np.zeros(
            (self.num_envs, self.capacity, *obs_shape), dtype=np.uint8
        )
        self.act_buf = np.zeros((self.num_envs, self.capacity), dtype=np.uint8)
        self.rew_buf = np.zeros((self.num_envs, self.capacity), dtype=np.float16)
        self.done_buf = np.zeros((self.num_envs, self.capacity), dtype=np.bool_)
        self.stoch_buf = np.zeros(
            (self.num_envs, self.capacity, config.latent_dim, config.num_classes),
            dtype=np.float16,
        )
        self.deter_buf = np.zeros(
            (self.num_envs, self.capacity, config.deter_dim), dtype=np.float16
        )
        self.positions = np.zeros(self.num_envs, dtype=np.int64)
        self.full = [False] * self.num_envs

    def store(self, obs, act, rew, done, stoch, deter):
        for env_idx in range(self.num_envs):
            pos = self.positions[env_idx]
            idx = pos % self.capacity

            self.obs_buf[env_idx, idx] = obs[env_idx]
            self.act_buf[env_idx, idx] = act[env_idx]
            self.rew_buf[env_idx, idx] = rew[env_idx].astype(np.float16)
            self.done_buf[env_idx, idx] = done[env_idx]
            self.stoch_buf[env_idx, idx] = (
                stoch[env_idx].cpu().numpy().astype(np.float16)
            )
            self.deter_buf[env_idx, idx] = (
                deter[env_idx].cpu().numpy().astype(np.float16)
            )

            self.positions[env_idx] += 1
            if self.positions[env_idx] >= self.capacity:
                self.full[env_idx] = True
                self.positions[env_idx] = 0

    def sample(self):
        # Sample one sequence from each environment's buffer
        indices = []
        start_indices = []
        for env_idx in range(self.num_envs):
            current_size = (
                self.capacity if self.full[env_idx] else self.positions[env_idx]
            )
            valid_end = current_size - self.sequence_length

            if valid_end <= 0:
                start = 0
            else:
                start = np.random.randint(0, valid_end)

            env_indices = (start + np.arange(self.sequence_length)) % self.capacity
            indices.append(env_indices)
            start_indices.append(start)

        # Stack indices across environments
        indices = np.stack(indices)

        return {
            "initial_stoch": torch.as_tensor(
                self.stoch_buf[np.arange(self.num_envs), start_indices],
                device=self.device,
                dtype=torch.float32,
            ),
            "initial_deter": torch.as_tensor(
                self.deter_buf[np.arange(self.num_envs), start_indices],
                device=self.device,
                dtype=torch.float32,
            ),
            "observation": torch.as_tensor(
                self.obs_buf[np.arange(self.num_envs)[:, None], indices],
                dtype=torch.float32,
                device=self.device,
            )
            .div_(255.0)
            .permute(1, 0, 2, 3, 4),
            "action": torch.as_tensor(
                self.act_buf[np.arange(self.num_envs)[:, None], indices],
                dtype=torch.long,
                device=self.device,
            ).permute(1, 0),
            "reward": torch.as_tensor(
                self.rew_buf[np.arange(self.num_envs)[:, None], indices],
                dtype=torch.float32,
                device=self.device,
            ).permute(1, 0),
            "done": torch.as_tensor(
                self.done_buf[np.arange(self.num_envs)[:, None], indices],
                dtype=torch.float32,
                device=self.device,
            ).permute(1, 0),
        }

    def __len__(self):
        # Return minimum available length across all environments
        return min(
            pos if not full else self.capacity
            for pos, full in zip(self.positions, self.full)
        )

    def size(self):
        # Return total size of the buffer
        return sum(
            pos if not full else self.capacity
            for pos, full in zip(self.positions, self.full)
        )


class LAProp(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-20):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.state["step"] = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["momentum_buffer"] = torch.zeros_like(p)

                exp_avg_sq = state["exp_avg_sq"]
                momentum_buffer = state["momentum_buffer"]

                # RMSProp update
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(eps)
                normalized_grad = grad / denom

                # Momentum update
                momentum_buffer.mul_(beta1).add_(normalized_grad, alpha=1 - beta1)

                # Parameter update
                p.add_(momentum_buffer, alpha=-group["lr"])

        return loss


class OneHotCategoricalStraightThrough(Categorical):
    def sample(self, sample_shape=torch.Size()):
        samples = super().sample(sample_shape)
        return self.probs + (samples - self.probs).detach()


class ObservationEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.apply(utils.init_weights)

    def forward(self, x):
        return torch.utils.checkpoint.checkpoint(self.conv, x)


class ObservationDecoder(nn.Module):
    def __init__(self, feature_dim, out_channels=3, output_size=(64, 64)):
        super().__init__()
        self.out_channels = out_channels
        self.output_size = output_size

        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256 * 8 * 8),
            nn.LayerNorm(256 * 8 * 8),
            nn.SiLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )
        self.apply(utils.init_weights)

    def forward(self, x):
        return torch.utils.checkpoint.checkpoint(self.net, x)


class TwoHotCategoricalStraightThrough(torch.distributions.Distribution):
    def __init__(self, logits, bins=255, low=-20.0, high=20.0):
        super().__init__(validate_args=False)
        self.logits = logits
        self.bin_centers = torch.linspace(low, high, bins, device=logits.device)

    def log_prob(self, value):
        value = utils.symlog(value).clamp(self.bin_centers[0], self.bin_centers[-1])
        indices = (
            (value - self.bin_centers[0]) / (self.bin_centers[1] - self.bin_centers[0])
        ).clamp(0, len(self.bin_centers) - 1)

        lower = indices.floor().long().unsqueeze(-1)
        upper = indices.ceil().long().unsqueeze(-1)
        alpha = (indices - lower.squeeze(-1)).unsqueeze(-1)

        probs = F.softmax(self.logits, dim=-1)
        return torch.log(
            (1 - alpha) * probs.gather(-1, lower) + alpha * probs.gather(-1, upper)
        ).squeeze(-1)

    @property
    def mean(self):
        return utils.symexp(
            (F.softmax(self.logits, dim=-1) * self.bin_centers).sum(-1, keepdim=True)
        )


class RSSM(nn.Module):
    def __init__(self, action_dim, latent_dim, num_classes, deter_dim, embed_dim):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.deter_dim = deter_dim
        self.embed_dim = embed_dim

        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, latent_dim * num_classes),
        )

        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, latent_dim * num_classes),
        )

        self.gru = nn.GRUCell(latent_dim * num_classes + action_dim, deter_dim)
        self.deter_init = nn.Parameter(torch.zeros(1, deter_dim))
        self.stoch_init = nn.Parameter(torch.zeros(1, latent_dim * num_classes))
        self.apply(utils.init_weights)

    def init_state(self, batch_size, device):
        stoch = (
            F.one_hot(
                torch.zeros(batch_size, self.latent_dim, dtype=torch.long),
                self.num_classes,
            )
            .float()
            .to(device)
        )
        deter = self.deter_init.repeat(batch_size, 1)
        return (stoch, deter)

    def imagine_step(self, stoch, deter, action):
        # Use prior network
        stoch = utils.symlog(stoch)
        action_oh = F.one_hot(action, self.action_dim).float()
        gru_input = torch.cat([stoch.flatten(1), action_oh], dim=1)
        deter = self.gru(gru_input, deter)
        if torch.isnan(deter).any():
            print("NaN detected in GRU input")
            deter = torch.nan_to_num(deter, nan=0.0)
        prior_logits = self.prior_net(deter).view(-1, self.latent_dim, self.num_classes)
        prior_logits = prior_logits - torch.logsumexp(prior_logits, -1, keepdim=True)
        prior_logits = torch.log(
            0.99 * torch.softmax(prior_logits, -1) + 0.01 / self.num_classes
        )
        stoch = F.gumbel_softmax(prior_logits, tau=1.0, hard=True)
        return utils.symexp(stoch), deter

    def observe_step(self, deter, embed):
        # Use posterior network
        post_logits = self.post_net(torch.cat([deter, embed], dim=1))
        post_logits = post_logits.view(-1, self.latent_dim, self.num_classes)
        post_logits = post_logits - torch.logsumexp(post_logits, -1, keepdim=True)
        post_logits = torch.log(
            0.99 * torch.softmax(post_logits, -1) + 0.01 / self.num_classes
        )
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
            gru_input = torch.cat([stoch.flatten(1), action_seq[t]], dim=1)
            deter = self.gru(gru_input, deter)

            prior_logits, post_logits = self.observe_step(deter, embed_seq[t])

            prior_dist = Independent(
                OneHotCategoricalStraightThrough(logits=prior_logits), 1
            )
            post_dist = Independent(
                OneHotCategoricalStraightThrough(logits=post_logits), 1
            )

            stoch = F.gumbel_softmax(post_logits, tau=1.0, hard=True)
            features.append(torch.cat([deter, stoch.flatten(1)], dim=1))

            priors.append(prior_dist)
            posteriors.append(post_dist)

        return (priors, posteriors), torch.stack(features)


class WorldModel(nn.Module):
    def __init__(
        self,
        in_channels,
        action_dim,
        embed_dim,
        latent_dim,
        num_classes,
        deter_dim,
        obs_size,
    ):
        super().__init__()
        self.encoder = ObservationEncoder(in_channels, embed_dim)
        self.rssm = RSSM(action_dim, latent_dim, num_classes, deter_dim, embed_dim)
        self.decoder = ObservationDecoder(
            deter_dim + latent_dim * num_classes, in_channels, obs_size[1:]
        )
        self.reward_decoder = nn.Sequential(
            nn.Linear(deter_dim + latent_dim * num_classes, 255)
        )
        self.continue_decoder = nn.Sequential(
            nn.Linear(deter_dim + latent_dim * num_classes, 1)
        )

        self.apply(utils.init_weights)
        self.reward_decoder[-1].weight.data.zero_()
        self.reward_decoder[-1].bias.data.zero_()

    def observe(self, observations, actions, stoch, deter):
        embed = self.encoder(observations.flatten(0, 1)).view(
            actions.size(0), actions.size(1), -1
        )
        actions_onehot = F.one_hot(actions, self.rssm.action_dim).float()

        priors, posteriors = [], []
        features = []

        for t in range(actions.size(0)):
            deter = self.rssm.gru(
                torch.cat([stoch.flatten(1), actions_onehot[t]], dim=1), deter
            )

            prior_logits = self.rssm.prior_net(deter).view(
                deter.size(0), self.rssm.latent_dim, self.rssm.num_classes
            )
            prior_dist = Independent(
                OneHotCategoricalStraightThrough(logits=prior_logits), 1
            )

            post_logits = self.rssm.post_net(torch.cat([deter, embed[t]], dim=1))
            post_logits = post_logits.view(
                deter.size(0), self.rssm.latent_dim, self.rssm.num_classes
            )
            post_dist = Independent(
                OneHotCategoricalStraightThrough(logits=post_logits), 1
            )

            stoch = F.gumbel_softmax(post_logits, tau=1.0, hard=True)
            features.append(torch.cat([deter, stoch.flatten(1)], dim=1))
            priors.append(prior_dist)
            posteriors.append(post_dist)

        features = torch.stack(features)
        recon_dist = self.decoder(features.flatten(0, 1))
        reward_dist = TwoHotCategoricalStraightThrough(
            self.reward_decoder(features.flatten(0, 1))
        )
        continue_pred = self.continue_decoder(features.flatten(0, 1))

        return (priors, posteriors), features, recon_dist, reward_dist, continue_pred


class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, action_dim),
        )
        self.apply(utils.init_weights)

    def forward(self, x):
        return Categorical(logits=self.net(x))


class Critic(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 255),
        )
        self.apply(utils.init_weights)
        # Initialize last layer to zeros as per the paper
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        return self.net(x)


class DreamerV3:
    def __init__(self, obs_shape, action_dim, config):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.config = config
        self.replay_buffer = ReplayBuffer(config, config.device, obs_shape)
        self.device = config.device
        self.num_envs = 16

        self.world_model = WorldModel(
            obs_shape[0],
            action_dim,
            config.embed_dim,
            config.latent_dim,
            config.num_classes,
            config.deter_dim,
            obs_shape,
        ).to(self.device)

        feature_dim = config.deter_dim + config.latent_dim * config.num_classes
        self.actor = Actor(feature_dim, action_dim).to(self.device)
        self.critic = Critic(feature_dim).to(self.device)
        self.target_critic = Critic(feature_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for param in self.target_critic.parameters():
            param.requires_grad = False

        self.bin_centers = torch.linspace(-20.0, 20.0, 255, device=self.device)

        self.optimizers = {
            "world": LAProp(
                self.world_model.parameters(),
                lr=config.lr,
                betas=(0.9, 0.99),
                eps=config.eps,
            ),
            "actor": LAProp(
                self.actor.parameters(),
                lr=config.actor_lr,
                betas=(0.9, 0.99),
                eps=config.eps,
            ),
            "critic": LAProp(
                self.critic.parameters(),
                lr=config.critic_lr,
                betas=(0.9, 0.99),
                eps=config.eps,
            ),
        }
        self.scalers = {
            "world": torch.amp.GradScaler("cuda"),
            "actor": torch.amp.GradScaler("cuda"),
            "critic": torch.amp.GradScaler("cuda"),
        }

        self.init_hidden_state()
        self._reset_stoch, self._reset_deter = self.world_model.rssm.init_state(
            self.num_envs, self.device
        )
        self.step = 0

    def init_hidden_state(self):
        self.hidden_state = self.world_model.rssm.init_state(self.num_envs, self.device)

    def reset_hidden_states(self, done_indices):
        """Reset hidden states for specified environment indices"""
        if not done_indices.any():
            return

        stoch, deter = self.hidden_state
        stoch[done_indices] = self._reset_stoch[done_indices]
        deter[done_indices] = self._reset_deter[done_indices]

    def act(self, observations):
        obs = torch.tensor(observations, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            stoch, deter = self.hidden_state
            embed = self.world_model.encoder(obs)

            # Get posteriors
            post_logits = self.world_model.rssm.observe_step(deter, embed)
            post_logits = post_logits.view(
                self.num_envs, self.config.latent_dim, self.config.num_classes
            )
            stoch = F.gumbel_softmax(post_logits, tau=1.0, hard=True)

            # Get actions
            feature = torch.cat([deter, stoch.flatten(1)], dim=1)
            action_dist = self.actor(feature)
            actions = action_dist.sample()

            # Update hidden states
            _, deter = self.world_model.rssm.imagine_step(stoch, deter, actions)
            self.hidden_state = (stoch, deter)

        return actions.cpu().numpy()

    def store_transition(self, obs, actions, rewards, dones):
        stoch, deter = self.hidden_state
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.device)
            quantized_obs = (obs_tensor * 255).clamp(0, 255).byte().cpu().numpy()
        self.replay_buffer.store(
            quantized_obs, actions, rewards, dones, stoch.detach(), deter.detach()
        )

    def update_world_model(self, batch):
        self.optimizers["world"].zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            init_stoch = batch["initial_stoch"]
            init_deter = batch["initial_deter"]
            obs, actions = batch["observation"], batch["action"]

            (priors, posteriors), features, recon_dist, reward_dist, continue_pred = (
                self.world_model.observe(obs, actions, init_stoch, init_deter)
            )

            prior_entropy = torch.stack([p.entropy() for p in priors]).mean()
            post_entropy = torch.stack([q.entropy() for q in posteriors]).mean()

            flat_feat = features.permute(1, 0, 2).flatten(0, 1)
            obs = batch["observation"]
            recon_target = obs.permute(1, 0, *range(2, obs.ndim)).flatten(0, 1)
            recon_pred = self.world_model.decoder(flat_feat)
            if obs.dtype == torch.uint8:
                recon_target = recon_target.float() / 255.0
            recon_loss = F.mse_loss(recon_pred, recon_target, reduction="mean")

            reward_loss = -reward_dist.log_prob(batch["reward"].flatten(0, 1)).mean()
            continue_loss = F.binary_cross_entropy_with_logits(
                continue_pred.flatten(0, 1), (1 - batch["done"].flatten(0, 1))
            )

            dyn_loss = torch.stack(
                [
                    torch.maximum(
                        torch.tensor(self.config.free_bits, device=self.device),
                        torch.distributions.kl_divergence(
                            Independent(  # Detach posterior logits
                                OneHotCategoricalStraightThrough(
                                    logits=posterior.base_dist.logits.detach()
                                ),
                                1,
                            ),
                            prior,
                        ).sum(dim=-1),
                    )
                    for prior, posterior in zip(priors, posteriors)
                ]
            ).mean()

            rep_loss = torch.stack(
                [
                    torch.maximum(
                        torch.tensor(self.config.free_bits, device=self.device),
                        torch.distributions.kl_divergence(
                            posterior,
                            Independent(  # Detach prior logits
                                OneHotCategoricalStraightThrough(
                                    logits=prior.base_dist.logits.detach()
                                ),
                                1,
                            ),
                        ).sum(dim=-1),
                    )
                    for prior, posterior in zip(priors, posteriors)
                ]
            ).mean()

            kl_loss = dyn_loss + rep_loss * self.config.rep_loss_scale
            total_loss = recon_loss + reward_loss + continue_loss + kl_loss

        self.scalers["world"].scale(total_loss).backward()
        self.scalers["world"].unscale_(self.optimizers["world"])
        utils.adaptive_gradient_clip(self.world_model, clip_factor=0.3, eps=1e-3)
        self.scalers["world"].step(self.optimizers["world"])
        self.scalers["world"].update()

        return {
            "world_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item(),
            "kl_loss": kl_loss.item(),
            "prior_entropy": prior_entropy.item(),
            "posterior_entropy": post_entropy.item(),
        }

    def update_actor_and_critic(self, replay_batch):
        B = self.config.batch_size
        init_state = (replay_batch["initial_stoch"], replay_batch["initial_deter"])

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            # Imagination rollout
            features, actions = self.world_model.rssm.imagine(
                init_state, self.actor, self.config.imagination_horizon
            )

            # Predict rewards and continues
            flat_features = features.flatten(0, 1)
            reward_logits = self.world_model.reward_decoder(flat_features)
            reward_dist = TwoHotCategoricalStraightThrough(reward_logits)
            rewards = utils.symlog(reward_dist.mean.view(features.shape[0], B))

            continue_pred = self.world_model.continue_decoder(flat_features)
            continues = continue_pred.view(features.shape[0], B)
            discounts = self.config.discount * continues

            # Compute values from target critic
            T, B, _ = features.shape
            critic_logits = self.target_critic(features.flatten(0, 1))
            probs = F.softmax(critic_logits, dim=-1)
            values = (probs * self.bin_centers).sum(-1)
            values = values.view(T, B)

            # Compute lambda returns
            lambda_returns = torch.zeros_like(values)
            lambda_returns[-1] = values[-1]
            for t in reversed(range(T - 1)):
                blended = (1 - self.config.gae_lambda) * values[
                    t
                ] + self.config.gae_lambda * lambda_returns[t + 1]
                lambda_returns[t] = rewards[t] + discounts[t] * blended

            # Return normalization
            returns_flat = lambda_returns.flatten()
            current_scale = torch.quantile(returns_flat, 0.95) - torch.quantile(
                returns_flat, 0.05
            )
            current_scale = current_scale.clamp(min=self.config.retnorm_limit)
            self.config.retnorm_scale = (
                self.config.retnorm_decay * self.config.retnorm_scale
                + (1 - self.config.retnorm_decay) * current_scale.item()
            )
            lambda_returns = lambda_returns / max(1.0, self.config.retnorm_scale)

            # Process replay buffer samples
            _, replay_features, _, _, _ = self.world_model.observe(
                replay_batch["observation"],
                replay_batch["action"],
                replay_batch["initial_stoch"],
                replay_batch["initial_deter"],
            )
            replay_rewards = replay_batch["reward"]
            replay_dones = replay_batch["done"]
            replay_continues = (1 - replay_dones.float()) * self.config.discount

            # Compute replay returns
            replay_values = (
                F.softmax(self.target_critic(replay_features.flatten(0, 1)), -1)
                * self.bin_centers
            ).sum(-1)
            replay_values = replay_values.view(replay_features.shape[0], B)

            replay_lambda_returns = torch.zeros_like(replay_values)
            replay_lambda_returns[-1] = replay_values[-1]
            for t in reversed(range(replay_features.shape[0] - 1)):
                blended = (1 - self.config.gae_lambda) * replay_values[
                    t
                ] + self.config.gae_lambda * replay_lambda_returns[t + 1]
                replay_lambda_returns[t] = (
                    replay_rewards[t] + replay_continues[t] * blended
                )

            # Normalize replay returns using the same scale
            replay_lambda_returns = replay_lambda_returns / max(
                1.0, self.config.retnorm_scale
            )

        # Critic update
        self.optimizers["critic"].zero_grad()

        # Imagination loss
        critic_logits = self.critic(features.flatten(0, 1))
        critic_dist = TwoHotCategoricalStraightThrough(critic_logits)
        imagination_loss = -critic_dist.log_prob(lambda_returns.flatten(0, 1)).mean()

        # Replay loss
        replay_critic_logits = self.critic(replay_features.flatten(0, 1))
        replay_critic_dist = TwoHotCategoricalStraightThrough(replay_critic_logits)
        replay_loss = -replay_critic_dist.log_prob(
            replay_lambda_returns.flatten(0, 1)
        ).mean()

        total_critic_loss = imagination_loss + 0.3 * replay_loss

        self.scalers["critic"].scale(total_critic_loss).backward()
        self.scalers["critic"].unscale_(self.optimizers["critic"])
        utils.adaptive_gradient_clip(self.critic, clip_factor=0.3, eps=1e-3)
        self.scalers["critic"].step(self.optimizers["critic"])
        self.scalers["critic"].update()

        # Update target critic
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            for online_param, target_param in zip(
                self.critic.parameters(), self.target_critic.parameters()
            ):
                target_param.data.mul_(self.config.critic_ema_decay).add_(
                    online_param.data, alpha=1 - self.config.critic_ema_decay
                )

        # Actor update
        self.optimizers["actor"].zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            online_probs = F.softmax(critic_logits, dim=-1)
            online_values = (
                (online_probs * self.bin_centers).sum(-1).view(features.shape[0], B)
            )
            online_values = online_values / max(1.0, self.config.retnorm_scale)
            advantages = (lambda_returns - online_values.detach()).flatten(0, 1)

            action_dist = self.actor(features.flatten(0, 1))
            log_probs = action_dist.log_prob(actions.flatten(0, 1))
            entropy = action_dist.entropy().mean()

            actor_loss = (
                -(log_probs * advantages.detach()).mean()
                - self.config.entropy_coef * entropy
            )

        self.scalers["actor"].scale(actor_loss).backward()
        self.scalers["actor"].unscale_(self.optimizers["actor"])
        utils.adaptive_gradient_clip(self.actor, clip_factor=0.3, eps=1e-3)
        self.scalers["actor"].step(self.optimizers["actor"])
        self.scalers["actor"].update()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": total_critic_loss.item(),
            "actor_entropy": entropy.item(),
        }

    def train(self):
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None

        losses = {
            "world_loss": 0,
            "recon_loss": 0,
            "reward_loss": 0,
            "continue_loss": 0,
            "kl_loss": 0,
            "actor_loss": 0,
            "critic_loss": 0,
            "actor_entropy": 0,
            "prior_entropy": 0,
            "posterior_entropy": 0,
        }

        for _ in range(self.config.updates_per_step):
            batch = self.replay_buffer.sample()

            # World model update
            wm_losses = self.update_world_model(batch)
            for k, v in wm_losses.items():
                losses[k] += v / self.config.updates_per_step

            # Actor-critic update
            ac_losses = self.update_actor_and_critic(batch)
            for k, v in ac_losses.items():
                losses[k] += v / self.config.updates_per_step

        self.step += 1
        return losses

    def save_checkpoint(self, env_name):
        os.makedirs("weights", exist_ok=True)
        torch.save(
            {
                "world_model": self.world_model.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            f"weights/{env_name}_dreamerv3.pt",
        )

    def load_checkpoint(self, env_name, mod="best"):
        checkpoint = torch.load(f"weights/{env_name}_{mod}_dreamerv3.pt")
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])


def train_dreamer(args):
    config = Config(args)

    step_counter = 0
    env = utils.make_vec_env(args.env, num_envs=16)
    env = utils.VideoLoggerWrapper(env, "videos", lambda: step_counter)

    obs_shape = env.single_observation_space.shape
    act_dim = env.single_action_space.n
    save_prefix = args.env.split("/")[-1]
    print(f"Env: {save_prefix}, Obs: {obs_shape}, Act: {act_dim}")

    agent = DreamerV3(obs_shape, act_dim, config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{save_prefix}_{timestamp}"
    utils.log_hparams(config, run_name)

    episode_history = []
    avg_reward_window = 100
    best_score, best_avg = float("-inf"), float("-inf")
    episode_scores = np.zeros(16)

    states, _ = env.reset()
    agent.init_hidden_state()

    while len(episode_history) < config.episodes:
        actions = agent.act(states)
        next_states, rewards, terms, truncs, _ = env.step(actions)
        dones = np.logical_or(terms, truncs)
        agent.store_transition(states, actions, rewards, dones)
        episode_scores += rewards

        reset_indices = np.where(dones)[0]
        if len(reset_indices) > 0:
            agent.reset_hidden_states(reset_indices)
            for idx in reset_indices:
                episode_history.append(episode_scores[idx])
                episode_scores[idx] = 0

        step_counter += 1
        states = next_states

        if len(agent.replay_buffer) >= config.min_buffer_size:
            if step_counter % config.update_interval == 0:
                losses = agent.train()
                utils.log_losses(step_counter, losses)

        avg_score = np.mean(episode_history[-avg_reward_window:])
        mem_size = agent.replay_buffer.size()
        utils.log_rewards(
            step_counter,
            avg_score,
            best_score,
            mem_size,
            len(episode_history),
            config.episodes,
        )

        if max(episode_history, default=float("-inf")) > best_score:
            best_score = max(episode_history)
            # agent.save_checkpoint(save_prefix + "_best")

        if avg_score > best_avg:
            best_avg = avg_score
            # agent.save_checkpoint(save_prefix + "_best_avg")

    print(f"\nFinished training. Best Avg.Score = {best_avg:.2f}")
    agent.save_checkpoint(save_prefix + "_final")
    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--wandb_key", type=str, default="../wandb.txt")
    args = parser.parse_args()
    for folder in ["videos", "weights"]:
        os.makedirs(folder, exist_ok=True)
    if args.env:
        train_dreamer(args)
    else:
        rand_order = np.random.permutation(ENV_LIST)
        for env in rand_order:
            args.env = env
            train_dreamer(args)
