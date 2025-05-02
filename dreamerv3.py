import os
import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from torch.distributions import Categorical, Independent
from torch.utils.tensorboard import SummaryWriter
from environment import AtariEnv, ENV_LIST
import utils
import warnings

warnings.simplefilter("ignore")
gym.register_envs(ale_py)
torch.backends.cudnn.benchmark = True


class Config:
    def __init__(self, args):
        self.capacity = 2000000
        self.batch_size = 16
        self.sequence_length = 64
        self.embed_dim = 1024
        self.latent_dim = 32
        self.num_classes = 32
        self.deter_dim = 4096
        self.hidden_dim = 512
        self.lr = 4e-5
        self.eps = 1e-5
        self.actor_lr = 4e-5
        self.critic_lr = 4e-5
        self.discount = 0.99
        self.gae_lambda = 0.95
        self.kl_scale = 1.0
        self.kl_balance = 0.8
        self.symlog_scale = 2.0
        self.imagination_horizon = 15
        self.min_buffer_size = 10000
        self.episodes = args.episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.free_bits = 1.0
        self.entropy_coef = 0.01
        self.updates_per_step = 5
        self.grad_clip_world = 1000.0
        self.grad_clip_ac = 100.0
        self.mixed_precision = True

    def to_dict(self):
        return {k: str(v) for k, v in vars(self).items()}


class ReplayBuffer:
    def __init__(self, config, device, obs_shape):
        self.capacity = config.capacity
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.symlog_scale = config.symlog_scale
        self.device = device
        self.obs_shape = obs_shape

        self.obs_buf = np.zeros((config.capacity, *obs_shape), dtype=np.uint8)
        self.act_buf = np.zeros(config.capacity, dtype=np.int64)
        self.rew_buf = np.zeros(config.capacity, dtype=np.float32)
        self.done_buf = np.zeros(config.capacity, dtype=bool)
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
        size = self.capacity if self.full else self.pos
        end = size - self.sequence_length
        starts = np.random.randint(0, end, size=self.batch_size)
        idxs = starts[:, None] + np.arange(self.sequence_length)

        obs_batch = torch.as_tensor(
            self.obs_buf[idxs], dtype=torch.float32, device=self.device
        ).div_(255.0)
        act_batch = torch.as_tensor(
            self.act_buf[idxs], dtype=torch.long, device=self.device
        )
        rew_batch = utils.symlog(
            torch.as_tensor(
                self.rew_buf[idxs], dtype=torch.float32, device=self.device
            ),
            scale=self.symlog_scale,
        )
        done_batch = torch.as_tensor(
            self.done_buf[idxs], dtype=torch.float32, device=self.device
        )

        return {
            "observation": obs_batch,
            "action": act_batch,
            "reward": rew_batch,
            "done": done_batch,
        }

    def __len__(self):
        return self.capacity if self.full else self.pos


class OneHotCategoricalStraightThrough(Categorical):
    def sample(self, sample_shape=torch.Size()):
        samples = super().sample(sample_shape)
        return self.probs + (samples - self.probs).detach()


class TwoHotCategoricalStraightThrough(torch.distributions.Distribution):
    def __init__(self, logits, bins=255, low=-20.0, high=20.0):
        super().__init__(validate_args=False)
        self.logits = logits
        self.bin_centers = torch.linspace(low, high, bins, device=logits.device)

    def log_prob(self, value_log):
        value_log = value_log.clamp(self.bin_centers[0], self.bin_centers[-1])

        step = self.bin_centers[1] - self.bin_centers[0]
        indices = (value_log - self.bin_centers[0]) / step

        lo = indices.floor().clamp(0, len(self.bin_centers) - 1).long()
        hi = indices.ceil().clamp(0, len(self.bin_centers) - 1).long()
        alpha = (indices - lo).unsqueeze(-1)

        probs = F.softmax(self.logits, dim=-1)
        p_lo = probs.gather(-1, lo.unsqueeze(-1))
        p_hi = probs.gather(-1, hi.unsqueeze(-1))

        mix = (1.0 - alpha) * p_lo + alpha * p_hi
        return mix.log().squeeze(-1)

    @property
    def mean(self):
        return utils.symexp(
            (F.softmax(self.logits, dim=-1) * self.bin_centers).sum(-1, keepdim=True)
        )

    def entropy(self):
        probs = F.softmax(self.logits, dim=-1)
        probs = probs.clamp(min=1e-12)
        return -(probs * torch.log(probs)).sum(dim=-1)


class ObservationEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.apply(utils.init_weights)

    def forward(self, x):
        return self.conv(x)


class ObservationDecoder(nn.Module):
    def __init__(self, feature_dim, out_channels, output_size):
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
            nn.Sigmoid(),  # bound [0,1]
        )
        self.apply(utils.init_weights)

    def forward(self, x):
        x = self.net(x)
        return x


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
            nn.LayerNorm(latent_dim * num_classes),
        )

        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, latent_dim * num_classes),
            nn.LayerNorm(latent_dim * num_classes),
        )

        self.gru = nn.GRUCell(latent_dim * num_classes + action_dim, deter_dim)
        self.deter_init = nn.Parameter(torch.zeros(1, deter_dim))
        self.stoch_init = nn.Parameter(torch.zeros(1, latent_dim * num_classes))
        self.apply(utils.init_weights)

    def init_state(self, batch_size, device):
        stoch = F.gumbel_softmax(
            self.stoch_init.repeat(batch_size, 1, 1), tau=1.0, hard=True
        )
        deter = self.deter_init.repeat(batch_size, 1)
        return (stoch, deter)

    def imagine_step(self, stoch, deter, action):
        # sample from prior
        action_oh = F.one_hot(action, self.action_dim).float()
        gru_input = torch.cat([stoch.flatten(1), action_oh], dim=1)
        deter = self.gru(gru_input, deter)

        prior_logits = self.prior_net(deter).view(-1, self.latent_dim, self.num_classes)
        stoch = F.gumbel_softmax(prior_logits, tau=1.0, hard=True)
        return stoch, deter

    def observe_step(self, deter, embed):
        # sample from posterior
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
        obs_size,
        action_dim,
        config,
    ):
        super().__init__()
        in_channels = obs_size[0]
        embed_dim = config.embed_dim
        deter_dim = config.deter_dim
        latent_dim = config.latent_dim
        num_classes = config.num_classes
        self.mixed_precision = config.mixed_precision
        self.encoder = ObservationEncoder(in_channels, embed_dim)
        self.rssm = RSSM(action_dim, latent_dim, num_classes, deter_dim, embed_dim)
        self.decoder = ObservationDecoder(
            deter_dim + latent_dim * num_classes, in_channels, obs_size[1:]
        )
        self.reward_decoder = nn.Sequential(
            nn.Linear(deter_dim + latent_dim * num_classes, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 255),
            nn.LayerNorm(255),
        )

        self.continue_decoder = nn.Sequential(
            nn.Linear(deter_dim + latent_dim * num_classes, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 1),
        )

    def observe(self, observations, actions):
        with torch.amp.autocast("cuda", enabled=self.mixed_precision):
            embed = self.encoder(observations.flatten(0, 1)).view(
                actions.size(0), actions.size(1), -1
            )
            actions_onehot = F.one_hot(actions, self.rssm.action_dim).float()

            priors, posteriors = [], []
            features = []
            stoch, deter = self.rssm.init_state(actions.size(1), observations.device)

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
            nn.Linear(512, 1),
        )
        self.apply(utils.init_weights)

    def forward(self, x):
        return self.net(x)


class DreamerV3:
    def __init__(self, obs_shape, action_dim, config):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.config = config
        self.replay_buffer = ReplayBuffer(config, config.device, obs_shape)
        self.device = config.device

        self.world_model = WorldModel(obs_shape, action_dim, config).to(self.device)

        feature_dim = config.deter_dim + config.latent_dim * config.num_classes
        self.actor = Actor(feature_dim, action_dim).to(self.device)
        self.critic = Critic(feature_dim).to(self.device)

        self.optimizers = {
            "world": optim.Adam(
                self.world_model.parameters(), lr=config.lr, eps=config.eps
            ),
            "actor": optim.Adam(self.actor.parameters(), lr=config.actor_lr),
            "critic": optim.Adam(self.critic.parameters(), lr=config.critic_lr),
        }
        self.scalers = {k: torch.amp.GradScaler("CUDA") for k in self.optimizers}

        self.hidden_state = None

    def init_hidden_state(self):
        self.hidden_state = self.world_model.rssm.init_state(1, self.device)

    def act(self, observation):
        obs = torch.tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            if self.hidden_state is None:
                self.init_hidden_state()

            stoch, deter = self.hidden_state
            embed = self.world_model.encoder(obs)

            # sample posterior
            post_logits = self.world_model.rssm.observe_step(deter, embed)
            post_logits = post_logits.view(
                1, self.config.latent_dim, self.config.num_classes
            )
            stoch = F.gumbel_softmax(post_logits, tau=1.0, hard=True)

            # choose action
            feature = torch.cat([deter, stoch.flatten(1)], dim=1)
            action = self.actor(feature).sample()

            # update hidden state
            _, deter = self.world_model.rssm.imagine_step(stoch, deter, action)
            self.hidden_state = (stoch, deter)

        return int(action.item())

    def store_transition(self, obs, action, reward, done):
        if not obs.dtype == torch.uint8:
            obs = utils.quantize(obs)
        self.replay_buffer.store(obs, action, reward, done)

    def _world_loss(self, obs, act, rew, done):
        (priors, posts), features, _, reward_dist, cont_pred = self.world_model.observe(
            obs, act
        )

        prior_entropy = torch.stack([p.entropy() for p in priors]).mean()
        post_entropy = torch.stack([q.entropy() for q in posts]).mean()

        feats_TBD = features.permute(1, 0, 2)
        flat_feat = feats_TBD.flatten(0, 1)

        obs_TB = obs.permute(1, 0, *range(2, obs.ndim))
        recon_pred = self.world_model.decoder(flat_feat)
        recon_target = obs_TB.flatten(0, 1)
        if obs.dtype == torch.uint8:
            recon_target = recon_target.float() / 255.0
        recon_loss = F.mse_loss(recon_pred, recon_target, reduction="mean")

        rew_TB = rew.permute(1, 0).flatten(0, 1)
        reward_loss = -reward_dist.log_prob(rew_TB).mean()
        reward_entropy = reward_dist.entropy().mean()

        cont_TB = cont_pred.flatten(0, 1)
        cont_target = (1 - done).permute(1, 0).flatten(0, 1)
        cont_loss = F.binary_cross_entropy_with_logits(cont_TB, cont_target)

        kl = torch.stack(
            [torch.distributions.kl_divergence(q, p) for p, q in zip(priors, posts)]
        )
        kl = kl.mean(0)  # average over time steps
        kl = torch.clamp_min(kl, self.config.free_bits / kl.shape[-1])
        kl = self.config.kl_balance * kl + (1 - self.config.kl_balance) * kl.detach()
        kl = kl.sum(-1)  # sum over latent dimensions
        kl_loss = kl.mean()

        total = recon_loss + reward_loss + cont_loss + self.config.kl_scale * kl_loss

        return (
            total,
            recon_loss,
            reward_loss,
            cont_loss,
            kl_loss,
            reward_entropy,
            prior_entropy,
            post_entropy,
        )

    def update_world_model(self):
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None

        opt = self.optimizers["world"]
        scaler = self.scalers["world"]
        opt.zero_grad()

        batch = self.replay_buffer.sample()
        obs = batch["observation"].to(self.device)
        act = batch["action"].to(self.device)
        rew = batch["reward"].to(self.device)
        done = batch["done"].to(self.device)

        with torch.amp.autocast("cuda", enabled=self.config.mixed_precision):
            total, r, rr, c, k, re, pre, poe = self._world_loss(obs, act, rew, done)

        scaler.scale(total).backward()
        torch.nn.utils.clip_grad_norm_(
            self.world_model.parameters(), self.config.grad_clip_world
        )
        scaler.step(opt)
        scaler.update()

        return {
            "world_loss": total.item(),
            "recon_loss": r.item(),
            "reward_loss": rr.item(),
            "continue_loss": c.item(),
            "kl_loss": k.item(),
            "reward_entropy": re.item(),
            "prior_entropy": pre.item(),
            "posterior_entropy": poe.item(),
        }

    def update_actor_and_critic(self):
        # imagination rollout
        with torch.no_grad(), torch.amp.autocast("cuda"):
            init_state = self.world_model.rssm.init_state(
                self.config.batch_size, self.device
            )
            feats, acts = self.world_model.rssm.imagine(
                init_state, self.actor, self.config.imagination_horizon
            )
            rd = self.world_model.reward_decoder(feats.flatten(0, 1))
            cd = self.world_model.continue_decoder(feats.flatten(0, 1))
            rewards = TwoHotCategoricalStraightThrough(rd)
            rewards = utils.symexp(
                rewards.mean, scale=self.config.symlog_scale
            ).view_as(acts)
            continues = torch.sigmoid(cd.view_as(acts)) * (
                1 - 1 / self.config.imagination_horizon
            )
            discounts = self.config.discount * continues

        flat = feats.reshape(-1, feats.shape[-1])

        # critic update
        opt_c = self.optimizers["critic"]
        scaler_c = self.scalers["critic"]
        opt_c.zero_grad()

        with torch.amp.autocast("cuda", enabled=self.config.mixed_precision):
            values = self.critic(flat).reshape_as(acts)
            last_value = values[-1]
            next_vals = torch.cat([values[1:], last_value.unsqueeze(0)], dim=0)
            deltas = rewards + discounts * next_vals - values
            weights = discounts * self.config.gae_lambda
            rev_deltas = deltas.flip(0)
            rev_weights = weights.flip(0)
            rev_prod = torch.cumprod(
                torch.cat([torch.ones_like(rev_weights[:1]), rev_weights[:-1]], 0),
                dim=0,
            )
            rev_adv = torch.cumsum(rev_deltas * rev_prod, dim=0)
            advantages = rev_adv.flip(0)
            returns = utils.symlog(advantages + rewards)
            critic_loss = F.mse_loss(values, returns.detach())

        scaler_c.scale(critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.grad_clip_ac
        )
        scaler_c.step(opt_c)
        scaler_c.update()

        # actor update
        opt_a = self.optimizers["actor"]
        scaler_a = self.scalers["actor"]
        opt_a.zero_grad()

        with torch.amp.autocast("cuda", enabled=self.config.mixed_precision):
            dist = self.actor(flat)
            logp = dist.log_prob(acts.reshape(-1))
            adv_flat = advantages.detach().reshape(-1)
            adv_norm = (adv_flat - adv_flat.mean()) / (
                adv_flat.std(unbiased=False) + 1e-8
            )
            entropy = dist.entropy().mean()
            actor_loss = -(logp * adv_norm).mean() - self.config.entropy_coef * entropy

        scaler_a.scale(actor_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.grad_clip_ac
        )
        scaler_a.step(opt_a)
        scaler_a.update()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_entropy": entropy.item(),
        }

    def train(self):
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None

        wm_losses = self.update_world_model()
        ac_losses = self.update_actor_and_critic()

        return {**wm_losses, **ac_losses}

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
    env = AtariEnv(args.env).make()
    obs_shape = env.observation_space.shape
    act_dim = env.action_space.n
    save_prefix = args.env.split("/")[-1]
    print(f"Env: {save_prefix}, Obs: {obs_shape}, Act: {act_dim}")

    config = Config(args)
    agent = DreamerV3(obs_shape, act_dim, config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"metrics/{save_prefix}_{timestamp}")

    writer.add_hparams(config.to_dict(), {})

    episode_history = []
    avg_reward_window = 100
    best_avg = float("-inf")
    score = 0

    state, _ = env.reset()
    agent.init_hidden_state()

    while len(episode_history) < config.episodes:
        action = agent.act(state)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        agent.store_transition(state, action, reward, done)
        score += reward

        if done:
            episode_history.append(score)
            ep = len(episode_history)

            if len(agent.replay_buffer) >= config.min_buffer_size:
                for _ in range(config.updates_per_step):
                    losses = agent.train()
                utils.log_losses(writer, ep, losses)

            avg_score = np.mean(episode_history[-avg_reward_window:])
            buffer_len = len(agent.replay_buffer)
            utils.log_rewards(writer, ep, score, avg_score, buffer_len, config.episodes)

            if score >= max(episode_history, default=-np.inf):
                agent.save_checkpoint(save_prefix + "_best")

            if avg_score >= best_avg:
                best_avg = avg_score
                agent.save_checkpoint(save_prefix + "_best_avg")

            score = 0
            agent.init_hidden_state()
            state, _ = env.reset()
        else:
            state = next_state

    print(f"\nFinished training. Final Avg.Score = {avg_score:.2f}")
    agent.save_checkpoint(save_prefix + "_final")
    writer.close()
    env.close()

    env = AtariEnv(args.env, shape=(64, 64), repeat=4, clip_rewards=False).make()
    utils.create_animation(env, agent, save_prefix, mod="best", seeds=10)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10000)
    args = parser.parse_args()

    for folder in ["metrics", "environments", "weights"]:
        os.makedirs(folder, exist_ok=True)

    if args.env:
        train_dreamer(args)
    else:
        rand_order = np.random.permutation(ENV_LIST)
        for env in rand_order:
            args.env = env
            train_dreamer(args)
