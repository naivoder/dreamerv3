import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import gymnasium as gym
import random
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
import imageio
import os
from datetime import datetime
import time
import warnings

warnings.simplefilter("ignore")


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"


def colored(text, color):
    return f"{color}{text}{Colors.ENDC}"


def progress_bar(current, total, width=20, label=""):
    percent = current / total
    filled = int(width * percent)
    bar = "▓" * filled + "░" * (width - filled)
    percentage = int(100 * percent)
    return f"{label} [{bar}] {percentage}%"


def spinning_cursor():
    while True:
        for cursor in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏":
            yield cursor


@dataclass
class Config:
    batch_size: int = 16
    sequence_length: int = 64
    replay_ratio: int = 64
    buffer_size: int = 1000000

    deter_size: int = 1024
    stoch_size: int = 16
    stoch_discrete: int = 16
    hidden_size: int = 256

    cnn_depth: int = 48
    cnn_kernels: List[int] = (4, 4, 4, 4)
    cnn_strides: List[int] = (2, 2, 2, 2)

    model_lr: float = 4e-5
    actor_lr: float = 4e-5
    critic_lr: float = 4e-5

    kl_weight: float = 1.0
    kl_balance: float = 1.0
    free_nats: float = 1.0
    pred_weight: float = 1.0
    dyn_weight: float = 1.0
    rep_weight: float = 0.1

    critic_weight: float = 1.0
    critic_replay_weight: float = 0.3

    gamma: float = 0.997
    lambda_: float = 0.95
    entropy_scale: float = 3e-4
    grad_clip_norm: float = 100.0
    weight_decay: float = 1e-6
    ema_decay: float = 0.98

    symlog_eps: float = 1e-8

    twohot_bins: int = 255
    twohot_min: float = -20.0
    twohot_max: float = 20.0

    unimix: float = 0.01

    horizon: int = 15

    return_norm_type: str = "percentile"
    return_norm_decay: float = 0.99
    return_norm_limit: float = 1.0


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def twohot_encode(
    x: torch.Tensor, bins: int, min_val: float, max_val: float
) -> torch.Tensor:
    x_symlog = symlog(x)
    x_symlog = torch.clamp(x_symlog, min_val, max_val)

    normalized = (x_symlog - min_val) / (max_val - min_val)
    scaled = normalized * (bins - 1)

    low = torch.floor(scaled).long()
    high = low + 1

    low = torch.clamp(low, 0, bins - 1)
    high = torch.clamp(high, 0, bins - 1)

    high_weight = scaled - low.float()
    low_weight = 1.0 - high_weight

    shape = list(x.shape) + [bins]
    twohot = torch.zeros(shape, device=x.device)

    batch_idx = torch.arange(x.numel(), device=x.device)
    twohot.view(-1, bins)[batch_idx, low.view(-1)] = low_weight.view(-1)
    twohot.view(-1, bins)[batch_idx, high.view(-1)] = high_weight.view(-1)

    return twohot


def twohot_decode(
    twohot: torch.Tensor, bins: int, min_val: float, max_val: float
) -> torch.Tensor:
    bin_centers = torch.linspace(min_val, max_val, bins, device=twohot.device)
    value_symlog = (twohot * bin_centers).sum(dim=-1)
    return symexp(value_symlog)


class OneHotDist(D.Distribution):
    arg_constraints = {}

    def __init__(self, logits: torch.Tensor, unimix: float = 0.01):
        super().__init__()
        uniform_logits = torch.zeros_like(logits)
        self.logits = (1 - unimix) * logits + unimix * uniform_logits
        self.probs = F.softmax(self.logits, dim=-1)
        self.cat = D.Categorical(logits=self.logits)

    def sample(self) -> torch.Tensor:
        indices = self.cat.sample()
        hard = F.one_hot(indices, self.logits.shape[-1]).float()
        soft = self.probs
        return hard - soft.detach() + soft

    def rsample(self) -> torch.Tensor:
        return self.sample()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        indices = torch.argmax(value, dim=-1)
        return self.cat.log_prob(indices)

    def entropy(self) -> torch.Tensor:
        return self.cat.entropy()

    @property
    def mode(self) -> torch.Tensor:
        indices = torch.argmax(self.logits, dim=-1)
        return F.one_hot(indices, self.logits.shape[-1]).float()

    def kl_divergence(self, other: "OneHotDist") -> torch.Tensor:
        return D.kl_divergence(self.cat, other.cat)


class CNNEncoder(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], config: Config):
        super().__init__()
        self.config = config

        if len(obs_shape) == 3:
            in_channels = obs_shape[-1]
        elif len(obs_shape) == 2:
            in_channels = 1
        else:
            raise ValueError(f"Unexpected observation shape: {obs_shape}")

        layers = []
        channels = [in_channels] + [
            config.cnn_depth * (2**i) for i in range(len(config.cnn_kernels))
        ]

        for i, (kernel, stride) in enumerate(
            zip(config.cnn_kernels, config.cnn_strides)
        ):
            layers.extend(
                [
                    nn.Conv2d(channels[i], channels[i + 1], kernel, stride),
                    nn.SiLU(),
                ]
            )

        self.cnn = nn.Sequential(*layers)

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, obs_shape[0], obs_shape[1])
            dummy_output = self.cnn(dummy_input)
            self.output_size = dummy_output.numel()

        self.fc = nn.Sequential(
            nn.Linear(self.output_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        elif len(obs.shape) == 4:
            obs = obs.permute(0, 3, 1, 2)

        x = self.cnn(obs)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class MLPEncoder(nn.Module):
    def __init__(self, obs_dim: int, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class RSSM(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], action_dim: int, config: Config):
        super().__init__()
        self.config = config
        self.obs_shape = obs_shape

        if len(obs_shape) > 1:
            self.encoder = CNNEncoder(obs_shape, config)
            self.is_image = True
        else:
            self.encoder = MLPEncoder(obs_shape[0], config)
            self.is_image = False

        gru_input_size = config.stoch_size * config.stoch_discrete + action_dim
        self.gru = nn.GRUCell(gru_input_size, config.deter_size)

        self.prior_net = nn.Sequential(
            nn.Linear(config.deter_size, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.stoch_size * config.stoch_discrete),
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(config.deter_size + config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.stoch_size * config.stoch_discrete),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if (
                module.out_features
                == self.config.stoch_size * self.config.stoch_discrete
            ):
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.xavier_normal_(module.weight, gain=1.0)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.GRUCell):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def initial_state(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        return {
            "deter": torch.zeros(batch_size, self.config.deter_size, device=device),
            "stoch": torch.zeros(
                batch_size,
                self.config.stoch_size,
                self.config.stoch_discrete,
                device=device,
            ),
        }

    def observe(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_state: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.is_image and len(obs.shape) == len(self.obs_shape) + 1:
            embed = self.encoder(obs)
        else:
            embed = self.encoder(obs)

        prev_stoch = prev_state["stoch"].reshape(prev_state["stoch"].shape[0], -1)
        deter = self.gru(
            torch.cat([prev_stoch, prev_action], dim=-1), prev_state["deter"]
        )

        prior_logits = self.prior_net(deter)
        prior_logits = prior_logits.reshape(
            -1, self.config.stoch_size, self.config.stoch_discrete
        )
        prior = OneHotDist(logits=prior_logits, unimix=self.config.unimix)

        posterior_logits = self.posterior_net(torch.cat([deter, embed], dim=-1))
        posterior_logits = posterior_logits.reshape(
            -1, self.config.stoch_size, self.config.stoch_discrete
        )
        posterior = OneHotDist(logits=posterior_logits, unimix=self.config.unimix)

        stoch = posterior.rsample()
        state = {"deter": deter, "stoch": stoch}

        prior_dist = prior.cat
        post_dist = posterior.cat
        kl_raw = D.kl_divergence(post_dist, prior_dist).sum(dim=-1)
        kl_balanced = (
            1 - self.config.kl_balance
        ) * kl_raw.detach() + self.config.kl_balance * kl_raw

        return state, prior, kl_balanced

    def imagine(
        self, prev_action: torch.Tensor, prev_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        prev_stoch = prev_state["stoch"].reshape(prev_state["stoch"].shape[0], -1)
        deter = self.gru(
            torch.cat([prev_stoch, prev_action], dim=-1), prev_state["deter"]
        )

        prior_logits = self.prior_net(deter)
        prior_logits = prior_logits.reshape(
            -1, self.config.stoch_size, self.config.stoch_discrete
        )
        prior = OneHotDist(logits=prior_logits, unimix=self.config.unimix)

        stoch = prior.rsample()

        return {"deter": deter, "stoch": stoch}


class Decoder(nn.Module):
    def __init__(self, state_dim: int, obs_shape: Tuple[int, ...], config: Config):
        super().__init__()
        self.config = config
        self.obs_shape = obs_shape
        self.is_image = len(obs_shape) > 1

        if self.is_image:
            self.obs_decoder = CNNDecoder(state_dim, obs_shape, config)
        else:
            self.obs_decoder = nn.Sequential(
                nn.Linear(state_dim, config.hidden_size),
                nn.SiLU(),
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.SiLU(),
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, obs_shape[0]),
            )

        self.reward_decoder = nn.Sequential(
            nn.Linear(state_dim, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.twohot_bins),
        )

        self.continue_decoder = nn.Sequential(
            nn.Linear(state_dim, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, 1),
        )

        self.apply(self._init_weights)

        nn.init.uniform_(self.reward_decoder[-1].weight, -0.001, 0.001)
        nn.init.zeros_(self.reward_decoder[-1].bias)
        nn.init.uniform_(self.continue_decoder[-1].weight, -0.001, 0.001)
        nn.init.zeros_(self.continue_decoder[-1].bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.get_feat(state)

        obs_pred = self.obs_decoder(feat)
        reward_logits = self.reward_decoder(feat)
        continue_logits = self.continue_decoder(feat)

        return obs_pred, reward_logits, continue_logits

    def get_feat(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        stoch = state["stoch"].reshape(state["stoch"].shape[0], -1)
        return torch.cat([state["deter"], stoch], dim=-1)


class CNNDecoder(nn.Module):
    def __init__(self, state_dim: int, obs_shape: Tuple[int, ...], config: Config):
        super().__init__()
        self.config = config
        self.obs_shape = obs_shape

        self.fc = nn.Sequential(
            nn.Linear(state_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, 4 * 4 * config.cnn_depth * 8),
            nn.SiLU(),
        )

        if len(obs_shape) == 3:
            out_channels = obs_shape[-1]
        else:
            out_channels = 1

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(config.cnn_depth * 8, config.cnn_depth * 4, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(config.cnn_depth * 4, config.cnn_depth * 2, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(config.cnn_depth * 2, config.cnn_depth, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(config.cnn_depth, out_channels, 4, 2, 1),
        )

        self.output_resize = nn.AdaptiveAvgPool2d((obs_shape[0], obs_shape[1]))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.fc(feat)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.deconv(x)
        x = self.output_resize(x)

        if len(self.obs_shape) == 3:
            x = x.permute(0, 2, 3, 1)
        else:
            x = x.squeeze(1)

        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)


class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, config: Config, discrete: bool = False
    ):
        super().__init__()
        self.config = config
        self.discrete = discrete

        self.net = nn.Sequential(
            nn.Linear(state_dim, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
        )

        if discrete:
            self.head = nn.Linear(config.hidden_size, action_dim)
        else:
            self.mean_head = nn.Linear(config.hidden_size, action_dim)
            self.std_head = nn.Linear(config.hidden_size, action_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(self, "head") and module is self.head:
                nn.init.xavier_uniform_(module.weight, gain=0.01)
            elif hasattr(self, "mean_head") and (
                module is self.mean_head or module is self.std_head
            ):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
            else:
                nn.init.xavier_normal_(module.weight, gain=1.0)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, feat: torch.Tensor) -> D.Distribution:
        h = self.net(feat)

        if self.discrete:
            logits = self.head(h)
            return OneHotDist(logits=logits, unimix=self.config.unimix)
        else:
            mean = self.mean_head(h)
            std = F.softplus(self.std_head(h) - 5) + 0.1
            return D.Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, state_dim: int, config: Config):
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Linear(state_dim, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.twohot_bins),
        )

        self.apply(self._init_weights)

        nn.init.uniform_(self.net[-1].weight, -0.001, 0.001)
        nn.init.zeros_(self.net[-1].bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class ReturnNormalizer:
    def __init__(self, decay: float = 0.99, limit: float = 1.0):
        self.decay = decay
        self.limit = limit
        self.scale = 1.0
        self.count = 0

    def update(self, returns: torch.Tensor):
        with torch.no_grad():
            self.count += 1
            if returns.numel() > 1:
                abs_returns = torch.abs(returns)
                percentile_5 = torch.quantile(abs_returns, 0.05)
                percentile_95 = torch.quantile(abs_returns, 0.95)
                scale = percentile_95 - percentile_5
                scale = torch.clamp(scale, min=1.0)
                if self.count > 10:
                    self.scale = (
                        self.decay * self.scale + (1 - self.decay) * scale.item()
                    )
                else:
                    self.scale = max(1.0, scale.item())

    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        return returns / max(self.limit, self.scale)


class WorldModel(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], action_dim: int, config: Config):
        super().__init__()
        self.config = config
        self.obs_shape = obs_shape

        self.rssm = RSSM(obs_shape, action_dim, config)
        self.decoder = Decoder(
            config.deter_size + config.stoch_size * config.stoch_discrete,
            obs_shape,
            config,
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        initial_state: Optional[Dict[str, torch.Tensor]] = None,
    ):
        batch_size, seq_len = obs.shape[:2]

        if initial_state is None:
            initial_state = self.rssm.initial_state(batch_size, obs.device)

        states = []
        priors = []
        kls = []

        state = initial_state
        for t in range(seq_len):
            if t == 0:
                prev_action = torch.zeros(
                    batch_size, action.shape[-1], device=obs.device
                )
            else:
                prev_action = action[:, t - 1]

            state, prior, kl = self.rssm.observe(obs[:, t], prev_action, state)
            states.append(state)
            priors.append(prior)
            kls.append(kl)

        states = {
            k: torch.stack([s[k] for s in states], dim=1) for k in states[0].keys()
        }
        kls = torch.stack(kls, dim=1)

        state_seq = {k: v.reshape(-1, *v.shape[2:]) for k, v in states.items()}
        obs_pred, reward_logits, continue_logits = self.decoder(state_seq)

        obs_pred = obs_pred.reshape(batch_size, seq_len, -1)
        reward_logits = reward_logits.reshape(batch_size, seq_len, -1)
        continue_logits = continue_logits.reshape(batch_size, seq_len, -1)

        return states, obs_pred, reward_logits, continue_logits, kls


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = {
            "obs": np.zeros((capacity,), dtype=object),
            "action": np.zeros((capacity,), dtype=object),
            "reward": np.zeros((capacity,), dtype=np.float32),
            "done": np.zeros((capacity,), dtype=np.float32),
        }
        self.idx = 0
        self.full = False

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool):
        self.data["obs"][self.idx] = obs
        self.data["action"][self.idx] = action
        self.data["reward"][self.idx] = reward
        self.data["done"][self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        max_idx = self.capacity if self.full else self.idx

        batch = {
            "obs": [],
            "action": [],
            "reward": [],
            "done": [],
        }

        for _ in range(batch_size):
            valid_start = False
            while not valid_start:
                start = random.randint(0, max_idx - seq_len)
                valid_start = True
                for i in range(seq_len - 1):
                    if self.data["done"][(start + i) % max_idx]:
                        valid_start = False
                        break

            for key in batch.keys():
                if key in ["obs", "action"]:
                    seq = np.array(
                        [self.data[key][(start + i) % max_idx] for i in range(seq_len)]
                    )
                else:
                    seq = self.data[key][start : start + seq_len]
                batch[key].append(seq)

        for key in batch.keys():
            batch[key] = torch.FloatTensor(np.stack(batch[key]))

        return batch

    def __len__(self):
        return self.capacity if self.full else self.idx


class DreamerAgent:
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        config: Config,
        discrete: bool = False,
    ):
        self.config = config
        self.discrete = discrete
        self.obs_shape = obs_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.world_model = WorldModel(obs_shape, action_dim, config).to(self.device)
        self.actor = Actor(
            config.deter_size + config.stoch_size * config.stoch_discrete,
            action_dim,
            config,
            discrete,
        ).to(self.device)
        self.critic = Critic(
            config.deter_size + config.stoch_size * config.stoch_discrete, config
        ).to(self.device)
        self.target_critic = Critic(
            config.deter_size + config.stoch_size * config.stoch_discrete, config
        ).to(self.device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()
        for param in self.target_critic.parameters():
            param.requires_grad = False

        self.world_opt = torch.optim.AdamW(
            self.world_model.parameters(),
            lr=config.model_lr,
            weight_decay=config.weight_decay,
            eps=1e-8,
        )
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(),
            lr=config.actor_lr,
            weight_decay=config.weight_decay,
            eps=1e-8,
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.weight_decay,
            eps=1e-8,
        )

        self.replay_buffer = ReplayBuffer(config.buffer_size)

        self.return_normalizer = ReturnNormalizer(
            decay=config.return_norm_decay, limit=config.return_norm_limit
        )

        self.train_steps = 0
        self.prev_action = None

    def process_observation(self, obs: np.ndarray) -> torch.Tensor:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        if len(self.obs_shape) > 1 and len(obs_t.shape) == 2:
            obs_t = obs_t.reshape(1, *self.obs_shape)

        return obs_t

    def act(
        self,
        obs: np.ndarray,
        state: Optional[Dict[str, torch.Tensor]] = None,
        training: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            obs_t = self.process_observation(obs)

            if state is None:
                state = self.world_model.rssm.initial_state(1, self.device)

            embed = self.world_model.rssm.encoder(obs_t)

            if self.prev_action is None:
                if self.discrete:
                    prev_action = torch.zeros(1, self.actor.head.out_features).to(
                        self.device
                    )
                else:
                    prev_action = torch.zeros(1, self.actor.mean_head.out_features).to(
                        self.device
                    )
            else:
                prev_action = (
                    torch.FloatTensor(self.prev_action).unsqueeze(0).to(self.device)
                )

            prev_stoch = state["stoch"].reshape(1, -1)
            deter = self.world_model.rssm.gru(
                torch.cat([prev_stoch, prev_action], dim=-1), state["deter"]
            )

            posterior_logits = self.world_model.rssm.posterior_net(
                torch.cat([deter, embed], dim=-1)
            )
            posterior_logits = posterior_logits.reshape(
                -1, self.config.stoch_size, self.config.stoch_discrete
            )
            posterior = OneHotDist(logits=posterior_logits, unimix=self.config.unimix)
            stoch = posterior.mode if not training else posterior.rsample()

            state = {"deter": deter, "stoch": stoch}

            feat = self.world_model.decoder.get_feat(state)
            action_dist = self.actor(feat)

            if training:
                action_t = action_dist.rsample()
            else:
                action_t = action_dist.mode if self.discrete else action_dist.mean

            action_np = action_t.cpu().numpy()[0]

            if self.discrete:
                action = np.argmax(action_np)
                action_onehot = np.zeros(self.actor.head.out_features)
                action_onehot[action] = 1.0
                self.prev_action = action_onehot
            else:
                action = np.clip(action_np, -1.0, 1.0)
                self.prev_action = action

            return action, state

    def train(self, steps: int = 1):
        if (
            len(self.replay_buffer)
            < self.config.batch_size * self.config.sequence_length
        ):
            return {}

        all_metrics = {}

        for _ in range(steps):
            batch = self.replay_buffer.sample(
                self.config.batch_size, self.config.sequence_length
            )
            for k, v in batch.items():
                batch[k] = v.to(self.device)

            world_metrics = self._train_world_model(batch)

            actor_critic_metrics = self._train_actor_critic(batch)

            metrics = {
                **world_metrics,
                **actor_critic_metrics,
                "buffer_size": len(self.replay_buffer),
                "train_steps": self.train_steps,
            }

            for k, v in metrics.items():
                if k in all_metrics:
                    all_metrics[k] += v
                else:
                    all_metrics[k] = v

            with torch.no_grad():
                for param, target_param in zip(
                    self.critic.parameters(), self.target_critic.parameters()
                ):
                    target_param.data = (
                        self.config.ema_decay * target_param.data
                        + (1 - self.config.ema_decay) * param.data
                    )

            self.train_steps += 1

        for k in all_metrics:
            all_metrics[k] /= steps

        return all_metrics

    def _adaptive_clip_grad(self, model: nn.Module, max_norm: float):
        for param in model.parameters():
            if param.grad is not None and len(param.shape) >= 2:
                weight_norm = param.data.norm()
                grad_norm = param.grad.data.norm()

                clip_coef = (0.1 * weight_norm / (grad_norm + 1e-6)).clamp(max=1.0)
                param.grad.data.mul_(clip_coef)

    def _train_world_model(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        done = batch["done"]

        states, obs_pred, reward_logits, continue_logits, kls = self.world_model(
            obs, action
        )

        obs_loss = F.mse_loss(
            obs_pred, obs.reshape(obs.shape[0], obs.shape[1], -1), reduction="mean"
        )

        reward_target = twohot_encode(
            reward,
            self.config.twohot_bins,
            self.config.twohot_min,
            self.config.twohot_max,
        )
        reward_loss = -torch.sum(
            reward_target * F.log_softmax(reward_logits, dim=-1), dim=-1
        ).mean()

        continue_loss = F.binary_cross_entropy_with_logits(
            continue_logits.squeeze(-1), 1 - done, reduction="mean"
        )

        kl_free = torch.maximum(
            kls, torch.tensor(self.config.free_nats, device=kls.device)
        )
        kl_loss = kl_free.mean()

        prediction_loss = self.config.pred_weight * (
            obs_loss + reward_loss + continue_loss
        )
        dynamics_loss = self.config.dyn_weight * kl_loss
        representation_loss = self.config.rep_weight * kl_loss

        loss = prediction_loss + dynamics_loss + representation_loss

        self.world_opt.zero_grad()
        loss.backward()

        self._adaptive_clip_grad(self.world_model, self.config.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(
            self.world_model.parameters(), self.config.grad_clip_norm
        )

        self.world_opt.step()

        with torch.no_grad():
            pred_rewards = twohot_decode(
                F.softmax(reward_logits, dim=-1),
                self.config.twohot_bins,
                self.config.twohot_min,
                self.config.twohot_max,
            )
            reward_error = F.mse_loss(pred_rewards, reward, reduction="mean")

            continue_acc = (
                ((torch.sigmoid(continue_logits.squeeze(-1)) > 0.5) == (1 - done))
                .float()
                .mean()
            )

            obs_error = F.mse_loss(
                obs_pred, obs.reshape(obs.shape[0], obs.shape[1], -1), reduction="mean"
            )

        return {
            "world/total_loss": loss.item(),
            "world/prediction_loss": prediction_loss.item(),
            "world/dynamics_loss": dynamics_loss.item(),
            "world/representation_loss": representation_loss.item(),
            "world/obs_loss": obs_loss.item(),
            "world/reward_loss": reward_loss.item(),
            "world/continue_loss": continue_loss.item(),
            "world/kl_loss": kl_loss.item(),
            "world/kl_mean": kls.mean().item(),
            "world/kl_max": kls.max().item(),
            "world/reward_error": reward_error.item(),
            "world/continue_accuracy": continue_acc.item(),
            "world/obs_error": obs_error.item(),
        }

    def _train_actor_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            obs = batch["obs"][:, 0]
            obs = self.process_observation(obs[0]).repeat(obs.shape[0], 1)
            if len(obs.shape) == 2 and len(self.obs_shape) > 1:
                obs = obs.reshape(obs.shape[0], *self.obs_shape)

            embed = self.world_model.rssm.encoder(obs)

            initial_state = self.world_model.rssm.initial_state(
                obs.shape[0], obs.device
            )
            posterior_logits = self.world_model.rssm.posterior_net(
                torch.cat([initial_state["deter"], embed], dim=-1)
            )
            posterior_logits = posterior_logits.reshape(
                -1, self.config.stoch_size, self.config.stoch_discrete
            )
            posterior = OneHotDist(logits=posterior_logits, unimix=self.config.unimix)
            stoch = posterior.rsample()

            initial_state = {"deter": initial_state["deter"], "stoch": stoch}

        critic_metrics = self._train_critic(initial_state, batch)
        actor_metrics = self._train_actor(initial_state)

        return {**critic_metrics, **actor_metrics}

    def _train_critic(
        self, initial_state: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        with torch.no_grad():
            state = {k: v.detach() for k, v in initial_state.items()}
            states = []
            rewards = []
            continues = []

            for _ in range(self.config.horizon):
                feat = self.world_model.decoder.get_feat(state)
                action_dist = self.actor(feat)
                action = action_dist.rsample()

                state = self.world_model.rssm.imagine(action, state)
                states.append({k: v.clone() for k, v in state.items()})

                _, reward_logits, continue_logits = self.world_model.decoder(state)

                reward_probs = F.softmax(reward_logits, dim=-1)
                reward = twohot_decode(
                    reward_probs,
                    self.config.twohot_bins,
                    self.config.twohot_min,
                    self.config.twohot_max,
                )
                rewards.append(reward)

                cont = torch.sigmoid(continue_logits).squeeze(-1)
                continues.append(cont)

            final_feat = self.world_model.decoder.get_feat(state)
            final_value_logits = self.target_critic(final_feat)
            final_value_probs = F.softmax(final_value_logits, dim=-1)
            final_value = twohot_decode(
                final_value_probs,
                self.config.twohot_bins,
                self.config.twohot_min,
                self.config.twohot_max,
            )

            rewards = torch.stack(rewards, dim=1)
            continues = torch.stack(continues, dim=1)

            values = []
            for state in states:
                feat = self.world_model.decoder.get_feat(state)
                value_logits = self.target_critic(feat)
                value_probs = F.softmax(value_logits, dim=-1)
                value = twohot_decode(
                    value_probs,
                    self.config.twohot_bins,
                    self.config.twohot_min,
                    self.config.twohot_max,
                )
                values.append(value)
            values = torch.stack(values, dim=1)

            bootstrap_values = torch.cat(
                [values[:, 1:], final_value.unsqueeze(1)], dim=1
            )
            returns = self._compute_lambda_returns(rewards, bootstrap_values, continues)

            self.return_normalizer.update(returns.flatten())

        imagination_loss = 0

        for t in range(self.config.horizon):
            feat = self.world_model.decoder.get_feat(states[t])
            value_logits = self.critic(feat)

            target_return = returns[:, t] / max(
                self.config.return_norm_limit, self.return_normalizer.scale
            )
            target_twohot = twohot_encode(
                target_return,
                self.config.twohot_bins,
                self.config.twohot_min,
                self.config.twohot_max,
            )

            loss = -torch.sum(
                target_twohot * F.log_softmax(value_logits, dim=-1), dim=-1
            )
            imagination_loss += loss.mean()

        imagination_loss = imagination_loss / self.config.horizon

        replay_loss = 0

        replay_states, _, _, _, _ = self.world_model(batch["obs"], batch["action"])

        num_replay_batches = min(self.config.sequence_length // 4, 8)

        for i in range(num_replay_batches):
            t = i * 4
            if t >= self.config.sequence_length:
                break

            with torch.no_grad():
                replay_state = {k: v[:, t].detach() for k, v in replay_states.items()}

                im_states = []
                im_rewards = []
                im_continues = []

                state = replay_state
                for h in range(
                    min(self.config.horizon, self.config.sequence_length - t)
                ):
                    feat = self.world_model.decoder.get_feat(state)
                    action_dist = self.actor(feat)
                    action = action_dist.rsample()

                    state = self.world_model.rssm.imagine(action, state)
                    im_states.append({k: v.clone() for k, v in state.items()})

                    _, reward_logits, continue_logits = self.world_model.decoder(state)
                    reward_probs = F.softmax(reward_logits, dim=-1)
                    reward = twohot_decode(
                        reward_probs,
                        self.config.twohot_bins,
                        self.config.twohot_min,
                        self.config.twohot_max,
                    )
                    im_rewards.append(reward)

                    cont = torch.sigmoid(continue_logits).squeeze(-1)
                    im_continues.append(cont)

                if len(im_rewards) > 0:
                    im_values = []
                    for s in im_states:
                        feat = self.world_model.decoder.get_feat(s)
                        value_logits = self.target_critic(feat)
                        value_probs = F.softmax(value_logits, dim=-1)
                        value = twohot_decode(
                            value_probs,
                            self.config.twohot_bins,
                            self.config.twohot_min,
                            self.config.twohot_max,
                        )
                        im_values.append(value)

                    final_feat = self.world_model.decoder.get_feat(state)
                    final_value_logits = self.target_critic(final_feat)
                    final_value_probs = F.softmax(final_value_logits, dim=-1)
                    final_value = twohot_decode(
                        final_value_probs,
                        self.config.twohot_bins,
                        self.config.twohot_min,
                        self.config.twohot_max,
                    )

                    im_rewards = torch.stack(im_rewards, dim=1)
                    im_continues = torch.stack(im_continues, dim=1)
                    im_values = torch.stack(im_values, dim=1)

                    bootstrap_values = torch.cat(
                        [im_values[:, 1:], final_value.unsqueeze(1)], dim=1
                    )
                    im_returns = self._compute_lambda_returns(
                        im_rewards, bootstrap_values, im_continues
                    )

                    replay_return = im_returns[:, 0] / max(
                        self.config.return_norm_limit, self.return_normalizer.scale
                    )

            replay_feat = self.world_model.decoder.get_feat(replay_state)
            replay_value_logits = self.critic(replay_feat)

            target_twohot = twohot_encode(
                replay_return,
                self.config.twohot_bins,
                self.config.twohot_min,
                self.config.twohot_max,
            )

            loss = -torch.sum(
                target_twohot * F.log_softmax(replay_value_logits, dim=-1), dim=-1
            )
            replay_loss += loss.mean()

        if num_replay_batches > 0:
            replay_loss = replay_loss / num_replay_batches

        critic_loss = (
            self.config.critic_weight * imagination_loss
            + self.config.critic_replay_weight * replay_loss
        )

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self._adaptive_clip_grad(self.critic, self.config.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.grad_clip_norm
        )
        self.critic_opt.step()

        with torch.no_grad():
            if len(states) > 0:
                sample_feat = self.world_model.decoder.get_feat(states[0])
                sample_value_logits = self.critic(sample_feat)
                sample_value_probs = F.softmax(sample_value_logits, dim=-1)
                sample_value = twohot_decode(
                    sample_value_probs,
                    self.config.twohot_bins,
                    self.config.twohot_min,
                    self.config.twohot_max,
                )
                mean_value_pred = sample_value.mean().item()
            else:
                mean_value_pred = 0.0

        return {
            "critic/loss": critic_loss.item(),
            "critic/imagination_loss": imagination_loss.item(),
            "critic/replay_loss": replay_loss.item() if num_replay_batches > 0 else 0.0,
            "critic/mean_return": returns.mean().item(),
            "critic/mean_return_normalized": (
                returns
                / max(self.config.return_norm_limit, self.return_normalizer.scale)
            )
            .mean()
            .item(),
            "critic/return_scale": self.return_normalizer.scale,
            "critic/mean_value_pred": mean_value_pred,
        }

    def _train_actor(self, initial_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
        states = []
        actions = []

        state = initial_state
        for _ in range(self.config.horizon):
            states.append(state)

            feat = self.world_model.decoder.get_feat(state)
            action_dist = self.actor(feat)
            action = action_dist.rsample()
            actions.append(action)

            state = self.world_model.rssm.imagine(action, state)

        with torch.no_grad():
            rewards = []
            continues = []
            values = []

            for state in states:
                _, reward_logits, continue_logits = self.world_model.decoder(state)

                reward_probs = F.softmax(reward_logits, dim=-1)
                reward = twohot_decode(
                    reward_probs,
                    self.config.twohot_bins,
                    self.config.twohot_min,
                    self.config.twohot_max,
                )
                rewards.append(reward)

                cont = torch.sigmoid(continue_logits).squeeze(-1)
                continues.append(cont)

                feat = self.world_model.decoder.get_feat(state)
                value_logits = self.critic(feat)
                value_probs = F.softmax(value_logits, dim=-1)
                value = twohot_decode(
                    value_probs,
                    self.config.twohot_bins,
                    self.config.twohot_min,
                    self.config.twohot_max,
                )
                values.append(value)

            final_feat = self.world_model.decoder.get_feat(state)
            final_value_logits = self.target_critic(final_feat)
            final_value_probs = F.softmax(final_value_logits, dim=-1)
            final_value = twohot_decode(
                final_value_probs,
                self.config.twohot_bins,
                self.config.twohot_min,
                self.config.twohot_max,
            )

            rewards = torch.stack(rewards, dim=1)
            continues = torch.stack(continues, dim=1)
            values_stacked = torch.stack(values, dim=1)

            bootstrap_values = torch.cat(
                [values_stacked[:, 1:], final_value.unsqueeze(1)], dim=1
            )
            returns = self._compute_lambda_returns(rewards, bootstrap_values, continues)

            scale = max(self.config.return_norm_limit, self.return_normalizer.scale)
            advantages = (returns - values_stacked) / scale

            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            advantages = torch.clamp(advantages, -10.0, 10.0)

        total_pg_loss = 0
        total_entropy = 0

        for t in range(self.config.horizon):
            feat = self.world_model.decoder.get_feat(states[t])
            action_dist = self.actor(feat)

            log_prob = action_dist.log_prob(actions[t])
            if len(log_prob.shape) > 1:
                log_prob = log_prob.sum(dim=-1)

            pg_loss = -(advantages[:, t].detach() * log_prob).mean()
            total_pg_loss += pg_loss

            entropy = action_dist.entropy()
            if len(entropy.shape) > 1:
                entropy = entropy.sum(dim=-1)
            total_entropy += entropy.mean()

        total_pg_loss = total_pg_loss / self.config.horizon
        total_entropy = total_entropy / self.config.horizon

        total_loss = total_pg_loss - self.config.entropy_scale * total_entropy

        self.actor_opt.zero_grad()
        total_loss.backward()
        self._adaptive_clip_grad(self.actor, self.config.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.grad_clip_norm
        )
        self.actor_opt.step()

        return {
            "actor/loss": total_loss.item(),
            "actor/pg_loss": total_pg_loss.item(),
            "actor/entropy": total_entropy.item(),
            "actor/mean_return": returns.mean().item(),
            "actor/mean_return_normalized": (returns / scale).mean().item(),
            "actor/mean_value": values_stacked.mean().item(),
            "actor/mean_advantage": advantages.mean().item(),
            "actor/advantage_std": advantages.std().item(),
            "actor/return_scale": self.return_normalizer.scale,
        }

    def _compute_lambda_returns(
        self, rewards: torch.Tensor, values: torch.Tensor, continues: torch.Tensor
    ) -> torch.Tensor:
        returns = torch.zeros_like(rewards)

        if rewards.shape[1] > 0:
            returns[:, -1] = (
                rewards[:, -1] + self.config.gamma * continues[:, -1] * values[:, -1]
            )

            for t in reversed(range(rewards.shape[1] - 1)):
                returns[:, t] = rewards[:, t] + self.config.gamma * continues[:, t] * (
                    (1 - self.config.lambda_) * values[:, t]
                    + self.config.lambda_ * returns[:, t + 1]
                )

        return returns


def train_dreamer(
    env_name: str, total_steps: int = 1000000
) -> Tuple[DreamerAgent, List[float]]:
    env = gym.make(env_name, render_mode=None)

    obs_shape = env.observation_space.shape
    is_image = len(obs_shape) > 1

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        discrete = True
    else:
        action_dim = env.action_space.shape[0]
        discrete = False

    config = Config()

    agent = DreamerAgent(obs_shape, action_dim, config, discrete)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("runs", exist_ok=True)
    env_name_safe = env_name.replace("/", "_")
    log_dir = f"runs/dreamer_v3_{env_name_safe}_{timestamp}"
    writer = SummaryWriter(log_dir)
    log_dir_msg = colored("📊 Logging to:", Colors.CYAN)
    print(f"{log_dir_msg} {log_dir}")

    config_header = colored("⚙️  Configuration:", Colors.BLUE)
    env_name_colored = colored(env_name, Colors.GREEN)
    action_type = "discrete" if discrete else "continuous"
    encoder_type = "CNN" if is_image else "MLP"

    print(f"\n{config_header}")
    print(f"  Environment: {env_name_colored}")
    print(f"  Observation shape: {obs_shape}, Action dim: {action_dim} ({action_type})")
    print(f"  Using {encoder_type} encoder")
    print(
        f"  Model: deter={config.deter_size}, stoch={config.stoch_size}×{config.stoch_discrete}"
    )
    print(
        f"  Learning rates: model={config.model_lr}, actor={config.actor_lr}, critic={config.critic_lr}"
    )
    print(f"  Replay ratio: {config.replay_ratio}")

    episode_rewards = []
    episode_count = 0
    total_env_steps = 0
    update_count = 0
    best_reward = -float("inf")

    warmup_steps = config.batch_size * config.sequence_length * 10

    spinner = spinning_cursor()
    start_time = time.time()

    os.makedirs("models", exist_ok=True)

    current_episode = {
        "obs": [],
        "action": [],
        "reward": [],
        "done": [],
    }

    obs, _ = env.reset()
    state = None
    episode_reward = 0
    agent.prev_action = None

    while total_env_steps < total_steps:
        action, state = agent.act(obs, state, training=True)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if discrete:
            action_onehot = np.zeros(action_dim)
            action_onehot[action] = 1
            agent.replay_buffer.add(obs, action_onehot, reward, float(done))
        else:
            agent.replay_buffer.add(obs, action, reward, float(done))

        episode_reward += reward
        obs = next_obs
        total_env_steps += 1

        if (
            total_env_steps > warmup_steps
            and total_env_steps
            % (config.batch_size * config.sequence_length // config.replay_ratio)
            == 0
        ):
            metrics = agent.train()
            update_count += 1

            for key, value in metrics.items():
                writer.add_scalar(f"train/{key}", value, total_env_steps)

            if update_count % 50 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = total_env_steps / elapsed
                eta = (
                    (total_steps - total_env_steps) / steps_per_sec
                    if steps_per_sec > 0
                    else 0
                )

                separator = colored("━" * 80, Colors.DIM)
                print(f"\n{separator}")

                env_header = colored(f"[{env_name}]", Colors.HEADER)
                update_num = colored(str(update_count), Colors.YELLOW)
                steps_info = colored(
                    f"{total_env_steps:,}/{total_steps:,}", Colors.YELLOW
                )
                speed_info = colored(f"{steps_per_sec:.1f}", Colors.YELLOW)
                eta_mins = int(eta // 60)
                eta_secs = int(eta % 60)
                eta_info = colored(f"{eta_mins}m {eta_secs}s", Colors.YELLOW)

                print(
                    f"{env_header} Update {update_num} | "
                    f"Steps {steps_info} | "
                    f"Speed {speed_info} steps/s | "
                    f"ETA {eta_info}"
                )

                world_loss = metrics.get("world/total_loss", 0)
                world_kl = metrics.get("world/kl_mean", 0)
                critic_loss = metrics.get("critic/loss", 0)
                actor_loss = metrics.get("actor/loss", 0)
                actor_entropy = metrics.get("actor/entropy", 0)

                losses_header = colored("🧠 Losses:", Colors.BLUE)
                world_loss_str = colored(f"{world_loss:.4f}", Colors.CYAN)
                world_kl_str = colored(f"{world_kl:.4f}", Colors.CYAN)
                critic_loss_str = colored(f"{critic_loss:.4f}", Colors.CYAN)
                actor_loss_str = colored(f"{actor_loss:.4f}", Colors.CYAN)
                actor_entropy_str = colored(f"{actor_entropy:.4f}", Colors.CYAN)

                print(
                    f"\n{losses_header} "
                    f"World: {world_loss_str}, "
                    f"KL: {world_kl_str}, "
                    f"Critic: {critic_loss_str}, "
                    f"Actor: {actor_loss_str}, "
                    f"Entropy: {actor_entropy_str}"
                )

                if episode_rewards:
                    recent_avg = np.mean(episode_rewards[-50:])
                    last_reward = episode_rewards[-1]
                    perf_header = colored("📈 Performance:", Colors.BLUE)
                    recent_str = colored(f"{recent_avg:.2f}", Colors.GREEN)
                    best_str = colored(f"{best_reward:.2f}", Colors.GREEN)
                    last_str = colored(f"{last_reward:.2f}", Colors.GREEN)

                    print(
                        f"{perf_header} "
                        f"Avg(50): {recent_str}, "
                        f"Best: {best_str}, "
                        f"Last: {last_str}"
                    )

        if done:
            episode_rewards.append(episode_reward)
            episode_count += 1

            writer.add_scalar("episode/reward", episode_reward, episode_count)

            if episode_count > 100 and episode_reward > best_reward:
                best_reward = episode_reward
                best_model_path = f"models/dreamer_v3_{env_name_safe}_best.pt"
                torch.save(
                    {
                        "world_model": agent.world_model.state_dict(),
                        "actor": agent.actor.state_dict(),
                        "critic": agent.critic.state_dict(),
                        "config": agent.config,
                        "episode": episode_count,
                        "reward": episode_reward,
                    },
                    best_model_path,
                )
                save_msg = colored("💾 New best model saved!", Colors.GREEN)
                reward_str = colored(f"{episode_reward:.2f}", Colors.BOLD)
                print(f"\n{save_msg} Reward: {reward_str}")

            if total_env_steps <= warmup_steps:
                warmup_label = colored("🔥 Warmup", Colors.YELLOW)
                reward_str = colored(f"{episode_reward:.2f}", Colors.CYAN)
                print(
                    f"\r{warmup_label} {next(spinner)} "
                    + f"Step {total_env_steps}/{warmup_steps}, "
                    + f"Episode {episode_count}, "
                    + f"Reward: {reward_str}",
                    end=" ",
                    flush=True,
                )

            obs, _ = env.reset()
            state = None
            episode_reward = 0
            agent.prev_action = None

    env.close()
    writer.close()

    complete_msg = colored("✅ Training Complete!", Colors.GREEN)
    time_elapsed = (time.time() - start_time) / 60
    time_str = colored(f"{time_elapsed:.1f} minutes", Colors.CYAN)

    print(f"\n{complete_msg}")
    print(f"Total time: {time_str}")

    return agent, episode_rewards


def evaluate_and_save_gif(agent: DreamerAgent, env_name: str, num_episodes: int = 5):
    env = gym.make(env_name, render_mode="rgb_array")

    best_reward = -float("inf")
    best_frames = []
    episode_rewards = []

    eval_header = colored(f"🎮 Evaluating {env_name}...", Colors.BLUE)
    print(f"\n{eval_header}")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        state = None
        episode_reward = 0
        frames = []
        episode_length = 0

        agent.prev_action = None

        while not done and episode_length < 1000:
            frames.append(env.render())

            action, state = agent.act(obs, state, training=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        reward_str = colored(f"{episode_reward:.2f}", Colors.CYAN)
        print(f"  Episode {ep+1}: Reward = {reward_str}, Length = {episode_length}")

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_frames = frames

    env.close()

    summary_header = colored("📊 Evaluation Summary:", Colors.GREEN)
    best_str = colored(f"{best_reward:.2f}", Colors.BOLD)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_str = colored(f"{mean_reward:.2f}", Colors.CYAN)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)

    print(f"\n{summary_header}")
    print(f"  Best: {best_str}")
    print(f"  Mean: {mean_str} ± {std_reward:.2f}")
    print(f"  Min/Max: {min_reward:.2f} / {max_reward:.2f}")

    if best_frames:
        os.makedirs("gifs", exist_ok=True)
        env_name_safe = env_name.replace("/", "_")
        gif_filename = f"gifs/dreamer_v3_{env_name_safe}.gif"
        imageio.mimsave(gif_filename, best_frames, fps=30)
        gif_msg = colored(gif_filename, Colors.GREEN)
        print(f"  Saved GIF: {gif_msg}")

    return episode_rewards


def main():
    environments = [
        "CartPole-v1",
        "Pendulum-v1",
        "BipedalWalker-v3",
        "LunarLander-v3",
        "CarRacing-v3",
        "Ant-v5",
        "ALE/Pong-v5",
    ]

    results = {}

    for env_name in environments:
        try:
            separator = "=" * 80
            header_msg = colored(f"🚀 Training Dreamer-v3 on {env_name}", Colors.HEADER)

            print(f"\n{separator}")
            print(f"{header_msg}")
            print(f"{separator}")

            total_steps = 1_000_000

            start_time = datetime.now()
            agent, rewards = train_dreamer(env_name, total_steps)
            train_time = (datetime.now() - start_time).total_seconds()

            env_name_safe = env_name.replace("/", "_")
            best_model_path = f"models/dreamer_v3_{env_name_safe}_best.pt"
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path)
                agent.world_model.load_state_dict(checkpoint["world_model"])
                agent.actor.load_state_dict(checkpoint["actor"])
                agent.critic.load_state_dict(checkpoint["critic"])
                load_msg = colored("📂 Loaded best model for evaluation", Colors.GREEN)
                best_ep = checkpoint.get("episode", "unknown")
                best_reward = checkpoint.get("reward", "unknown")
                print(f"\n{load_msg}")
                print(f"  Episode: {best_ep}, Reward: {best_reward}")

            eval_rewards = evaluate_and_save_gif(agent, env_name)

            results[env_name] = {
                "train_rewards": rewards,
                "eval_rewards": eval_rewards,
                "train_time": train_time,
                "final_avg_reward": (
                    np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                ),
            }

        except Exception as e:
            error_msg = colored(f"❌ Error training on {env_name}:", Colors.RED)
            print(f"\n{error_msg} {e}")
            import traceback

            traceback.print_exc()
            continue

    separator = "=" * 80
    final_header = colored("✅ TRAINING COMPLETE - FINAL SUMMARY", Colors.HEADER)

    print(f"\n{separator}")
    print(f"{final_header}")
    print(f"{separator}")

    for env_name, res in results.items():
        env_colored = colored(env_name, Colors.BLUE)
        train_time_mins = res["train_time"] / 60
        final_avg = res["final_avg_reward"]
        final_avg_str = colored(f"{final_avg:.2f}", Colors.GREEN)
        eval_mean = np.mean(res["eval_rewards"])
        eval_std = np.std(res["eval_rewards"])
        eval_mean_str = colored(f"{eval_mean:.2f}", Colors.CYAN)
        num_episodes = len(res["train_rewards"])

        print(f"\n{env_colored}:")
        print(f"  Training Time: {train_time_mins:.1f} minutes")
        print(f"  Final Training Avg: {final_avg_str}")
        print(f"  Eval Mean: {eval_mean_str} ± {eval_std:.2f}")
        print(f"  Episodes: {num_episodes}")


if __name__ == "__main__":
    main()
