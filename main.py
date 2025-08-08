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
import traceback
from collections import deque
from gymnasium.vector import AsyncVectorEnv
import torch._dynamo
import logging

warnings.simplefilter("ignore")
logging.getLogger("torch").setLevel(logging.CRITICAL)
os.environ["TRITON_LOG_LEVEL"] = "0"
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


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
    replay_ratio: int = 32
    buffer_size: int = 5_000_000
    deter_size: int = 1024
    stoch_size: int = 16
    stoch_discrete: int = 16
    hidden_size: int = 256
    cnn_depth: int = 48
    cnn_kernels: List[int] = (4, 4, 4, 4)
    cnn_strides: List[int] = (2, 2, 2, 2)
    learning_rate: float = 4e-5
    free_nats: float = 1.0
    pred_weight: float = 1.0
    dyn_weight: float = 1.0
    rep_weight: float = 0.1
    critic_weight: float = 1.0
    critic_replay_weight: float = 0.3
    gamma: float = 0.997
    lambda_: float = 0.95
    entropy_scale: float = 3e-4
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
    action_repeat: int = 1
    online_fraction: float = 0.5
    agc_clip_factor: float = 0.3
    agc_eps: float = 1e-3
    laprop_eps: float = 1e-20
    num_envs: int = 16


@torch.jit.script
def symlog(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * (x.abs() + 1).log()


@torch.jit.script
def symexp(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * (x.abs().exp() - 1)


@torch.jit.script
def twohot_encode_jit(
    x: torch.Tensor, bins: int, min_val: float, max_val: float
) -> torch.Tensor:
    x_symlog = symlog(x)
    x_symlog = x_symlog.clamp(min_val, max_val)

    normalized = (x_symlog - min_val) / (max_val - min_val)
    scaled = normalized * (bins - 1)

    low = scaled.floor().long()
    high = (low + 1).clamp(0, bins - 1)
    low = low.clamp(0, bins - 1)

    high_weight = scaled - low.float()
    low_weight = 1.0 - high_weight

    shape = list(x.shape) + [bins]
    twohot = torch.zeros(shape, device=x.device, dtype=x.dtype)

    indices = torch.arange(x.numel(), device=x.device)
    twohot_flat = twohot.view(-1, bins)

    twohot_flat[indices, low.view(-1)] = low_weight.view(-1)
    twohot_flat[indices, high.view(-1)] = high_weight.view(-1)

    return twohot


def twohot_encode(
    x: torch.Tensor, bins: int, min_val: float, max_val: float
) -> torch.Tensor:
    return twohot_encode_jit(x, bins, min_val, max_val)


@torch.jit.script
def twohot_decode_jit(
    twohot: torch.Tensor, bins: int, min_val: float, max_val: float
) -> torch.Tensor:
    bin_centers = torch.linspace(min_val, max_val, bins, device=twohot.device)

    positive_mask = bin_centers >= 0
    negative_mask = ~positive_mask

    value_symlog = (twohot * bin_centers).sum(dim=-1)
    return symexp(value_symlog)


def twohot_decode(
    twohot: torch.Tensor, bins: int, min_val: float, max_val: float
) -> torch.Tensor:
    return twohot_decode_jit(twohot, bins, min_val, max_val)


class LaProp(torch.optim.Optimizer):
    def __init__(self, params, lr=4e-5, betas=(0.9, 0.99), eps=1e-20):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["v"] = torch.zeros_like(p)
                    state["m"] = torch.zeros_like(p)

                v, m = state["v"], state["m"]
                state["step"] += 1

                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                normalized_grad = grad / (v.sqrt() + eps)
                m.mul_(beta1).add_(normalized_grad, alpha=1 - beta1)
                p.add_(m, alpha=-lr)

        return loss


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).sqrt()
        return x / (norm + self.eps) * self.scale


@torch.jit.script
def adaptive_gradient_clip(
    grad: torch.Tensor,
    weight: torch.Tensor,
    clip_factor: float = 0.3,
    eps: float = 1e-3,
) -> torch.Tensor:
    if weight.numel() < 2:
        return grad

    grad_norm = grad.norm(2)
    weight_norm = weight.norm(2)
    max_norm = weight_norm * clip_factor

    if grad_norm > max_norm:
        return grad * (max_norm / (grad_norm + eps))
    return grad


def adaptive_gradient_clip_model(
    model: nn.Module, clip_factor: float = 0.3, eps: float = 1e-3
):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data = adaptive_gradient_clip(p.grad.data, p.data, clip_factor, eps)


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


class BlockGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_blocks: int = 8):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks

        self.input_proj = nn.Linear(input_size + hidden_size, 3 * hidden_size)

        self.block_weights = nn.Parameter(
            torch.zeros(num_blocks, self.block_size, 3 * self.block_size)
        )
        self.block_biases = nn.Parameter(torch.zeros(num_blocks, 3 * self.block_size))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        for i in range(self.num_blocks):
            nn.init.xavier_uniform_(self.block_weights[i])
            nn.init.zeros_(self.block_biases[i])

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        input_gates = self.input_proj(torch.cat([x, h], dim=-1))
        input_gates = input_gates.view(batch_size, self.num_blocks, 3 * self.block_size)

        h_blocks = h.view(batch_size, self.num_blocks, self.block_size)

        block_gates = torch.bmm(
            h_blocks.reshape(-1, 1, self.block_size),
            self.block_weights.repeat(batch_size, 1, 1),
        ).squeeze(1)
        block_gates = block_gates.view(batch_size, self.num_blocks, -1)
        block_gates = input_gates + block_gates + self.block_biases

        r, z, n = block_gates.chunk(3, dim=-1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(n)

        new_h = (1 - z) * n + z * h_blocks

        return new_h.reshape(batch_size, self.hidden_size)


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
                    nn.SiLU(inplace=True),
                ]
            )

        self.cnn = nn.Sequential(*layers)

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, obs_shape[0], obs_shape[1])
            dummy_output = self.cnn(dummy_input)
            self.output_size = dummy_output.numel()

        self.fc = nn.Sequential(
            nn.Linear(self.output_size, config.hidden_size),
            RMSNorm(config.hidden_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs / 255.0 if obs.max() > 1.0 else obs

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
            nn.SiLU(inplace=True),
            RMSNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(inplace=True),
            RMSNorm(config.hidden_size),
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
        self.gru = BlockGRU(gru_input_size, config.deter_size, num_blocks=8)

        self.prior_net = nn.Sequential(
            nn.Linear(config.deter_size, config.hidden_size),
            nn.SiLU(inplace=True),
            RMSNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.stoch_size * config.stoch_discrete),
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(config.deter_size + config.hidden_size, config.hidden_size),
            nn.SiLU(inplace=True),
            RMSNorm(config.hidden_size),
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

        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.scale)

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

        batch_size = prior_logits.shape[0]

        prior_no_grad = OneHotDist(
            logits=prior_logits.detach(), unimix=self.config.unimix
        )
        posterior_no_grad = OneHotDist(
            logits=posterior_logits.detach(), unimix=self.config.unimix
        )

        prior_independent = D.Independent(prior.cat, 1)
        prior_no_grad_independent = D.Independent(prior_no_grad.cat, 1)
        posterior_independent = D.Independent(posterior.cat, 1)
        posterior_no_grad_independent = D.Independent(posterior_no_grad.cat, 1)

        kl_dyn_raw = D.kl_divergence(posterior_no_grad_independent, prior_independent)

        kl_rep_raw = D.kl_divergence(posterior_independent, prior_no_grad_independent)

        kl_dyn = torch.maximum(kl_dyn_raw, torch.tensor(self.config.free_nats))
        kl_rep = torch.maximum(kl_rep_raw, torch.tensor(self.config.free_nats))

        return state, prior, kl_dyn, kl_rep, kl_dyn_raw, kl_rep_raw

    def imagine(
        self, prev_action: torch.Tensor, prev_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        prev_stoch = prev_state["stoch"].view(prev_state["stoch"].shape[0], -1)
        deter = self.gru(
            torch.cat([prev_stoch, prev_action], dim=-1), prev_state["deter"]
        )

        prior_logits = self.prior_net(deter)
        prior_logits = prior_logits.view(
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
                nn.SiLU(inplace=True),
                RMSNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.SiLU(inplace=True),
                RMSNorm(config.hidden_size),
                nn.Linear(config.hidden_size, obs_shape[0]),
            )

        self.reward_decoder = nn.Sequential(
            nn.Linear(state_dim, config.hidden_size),
            nn.SiLU(inplace=True),
            RMSNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(inplace=True),
            RMSNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.twohot_bins),
        )

        self.continue_decoder = nn.Sequential(
            nn.Linear(state_dim, config.hidden_size),
            nn.SiLU(inplace=True),
            RMSNorm(config.hidden_size),
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

        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.scale)

    def forward(
        self, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.get_feat(state)

        obs_pred = self.obs_decoder(feat)
        reward_logits = self.reward_decoder(feat)
        continue_logits = self.continue_decoder(feat)

        return obs_pred, reward_logits, continue_logits

    def get_feat(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        stoch = state["stoch"].view(state["stoch"].shape[0], -1)
        return torch.cat([state["deter"], stoch], dim=-1)


class CNNDecoder(nn.Module):
    def __init__(self, state_dim: int, obs_shape: Tuple[int, ...], config: Config):
        super().__init__()
        self.config = config
        self.obs_shape = obs_shape

        self.fc = nn.Sequential(
            nn.Linear(state_dim, config.hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_size, 4 * 4 * config.cnn_depth * 8),
            nn.SiLU(inplace=True),
        )

        if len(obs_shape) == 3:
            out_channels = obs_shape[-1]
        else:
            out_channels = 1

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(config.cnn_depth * 8, config.cnn_depth * 4, 4, 2, 1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(config.cnn_depth * 4, config.cnn_depth * 2, 4, 2, 1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(config.cnn_depth * 2, config.cnn_depth, 4, 2, 1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(config.cnn_depth, out_channels, 4, 2, 1),
            nn.Sigmoid(),
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
        self,
        state_dim: int,
        action_dim: int,
        config: Config,
        discrete: bool = False,
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.config = config
        self.discrete = discrete
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = nn.Sequential(
            nn.Linear(state_dim, config.hidden_size),
            nn.SiLU(inplace=True),
            RMSNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(inplace=True),
            RMSNorm(config.hidden_size),
        )

        if discrete:
            self.head = nn.Linear(config.hidden_size, action_dim)
        else:
            self.head = nn.Linear(config.hidden_size, action_dim * 2)

            if action_low is not None and action_high is not None:
                self.register_buffer(
                    "action_scale",
                    torch.tensor(
                        (action_high - action_low) / 2.0,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )
                self.register_buffer(
                    "action_bias",
                    torch.tensor(
                        (action_high + action_low) / 2.0,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )
            else:
                self.register_buffer(
                    "action_scale", torch.ones(action_dim, device=self.device)
                )
                self.register_buffer(
                    "action_bias", torch.zeros(action_dim, device=self.device)
                )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(self, "head") and module is self.head:
                nn.init.xavier_uniform_(module.weight, gain=0.01)
            else:
                nn.init.xavier_normal_(module.weight, gain=1.0)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, feat: torch.Tensor, training: bool = False):
        h = self.net(feat)

        if self.discrete:
            logits = self.head(h)
            dist = OneHotDist(logits=logits, unimix=self.config.unimix)
            if training:
                action = dist.rsample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                action_indices = torch.argmax(action, dim=-1)
                return action_indices, log_prob, entropy
            else:
                return torch.argmax(dist.mode, dim=-1)
        else:
            output = self.head(h)
            mean, log_std = output.chunk(2, dim=-1)

            # Constrain log_std to reasonable range
            log_std_min, log_std_max = -5.0, 2.0
            log_std = log_std_min + (log_std_max - log_std_min) / 2 * (
                torch.tanh(log_std) + 1
            )
            std = torch.exp(log_std)

            distribution = D.Normal(mean, std)

            if training:
                sample = distribution.rsample()
                sample_tanh = torch.tanh(sample)
                action = sample_tanh * self.action_scale + self.action_bias

                log_prob = distribution.log_prob(sample)
                # Correction term: -log|det(d(tanh)/dx)| = -log(1 - tanh^2(x))
                log_prob -= torch.log(
                    self.action_scale * (1 - sample_tanh.pow(2)) + 1e-6
                )
                log_prob = log_prob.sum(dim=-1)  # Sum over action dimensions
                entropy = distribution.entropy().sum(dim=-1)

                return action, log_prob, entropy
            else:
                # For evaluation, use mean action
                mean_tanh = torch.tanh(mean)
                action = mean_tanh * self.action_scale + self.action_bias
                return action


class Critic(nn.Module):
    def __init__(self, state_dim: int, config: Config):
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Linear(state_dim, config.hidden_size),
            nn.SiLU(inplace=True),
            RMSNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(inplace=True),
            RMSNorm(config.hidden_size),
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

        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.scale)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class ReturnNormalizer(nn.Module):
    def __init__(
        self, device, decay=0.99, limit=1.0, percentile_low=0.05, percentile_high=0.95
    ):
        super().__init__()
        self.decay = decay
        self.limit = limit
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high

        self.register_buffer("low", torch.zeros((), dtype=torch.float32, device=device))
        self.register_buffer(
            "high", torch.zeros((), dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "scale", torch.ones((), dtype=torch.float32, device=device)
        )

    def update(self, values: torch.Tensor):
        with torch.no_grad():
            values = values.detach().flatten()

            if values.numel() > 1:
                low = torch.quantile(values, self.percentile_low)
                high = torch.quantile(values, self.percentile_high)

                self.low = self.decay * self.low + (1 - self.decay) * low
                self.high = self.decay * self.high + (1 - self.decay) * high

                self.scale = torch.max(
                    torch.tensor(self.limit, device=values.device), self.high - self.low
                )


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
        kl_dyn_list = []
        dyn_raw_list = []
        kl_rep_list = []
        rep_raw_list = []

        state = initial_state
        for t in range(seq_len):
            if t == 0:
                prev_action = torch.zeros(
                    batch_size, action.shape[-1], device=obs.device
                )
            else:
                prev_action = action[:, t - 1]

            state, _, kl_dyn, kl_rep, kl_dyn_raw, kl_rep_raw = self.rssm.observe(
                obs[:, t], prev_action, state
            )
            states.append(state)
            kl_dyn_list.append(kl_dyn)
            dyn_raw_list.append(kl_dyn_raw)
            kl_rep_list.append(kl_rep)
            rep_raw_list.append(kl_rep_raw)

        states = {
            k: torch.stack([s[k] for s in states], dim=1) for k in states[0].keys()
        }
        kl_dyns = torch.stack(kl_dyn_list, dim=1)
        kl_reps = torch.stack(kl_rep_list, dim=1)
        kl_dyns_raw = torch.stack(dyn_raw_list, dim=1)
        kl_reps_raw = torch.stack(rep_raw_list, dim=1)

        state_seq = {k: v.view(-1, *v.shape[2:]) for k, v in states.items()}
        obs_pred, reward_logits, continue_logits = self.decoder(state_seq)

        obs_pred = obs_pred.view(batch_size, seq_len, -1)
        reward_logits = reward_logits.view(batch_size, seq_len, -1)
        continue_logits = continue_logits.view(batch_size, seq_len, -1)

        return (
            states,
            obs_pred,
            reward_logits,
            continue_logits,
            kl_dyns,
            kl_reps,
            kl_dyns_raw,
            kl_reps_raw,
        )


class ParallelReplayBuffer:
    def __init__(self, capacity: int, num_envs: int):
        self.capacity = capacity // num_envs
        self.num_envs = num_envs
        self.data = {
            "obs": np.zeros((num_envs, self.capacity), dtype=object),
            "action": np.zeros((num_envs, self.capacity), dtype=object),
            "reward": np.zeros((num_envs, self.capacity), dtype=np.float32),
            "done": np.zeros((num_envs, self.capacity), dtype=np.float32),
            "state": np.zeros((num_envs, self.capacity), dtype=object),
        }
        self.idx = np.zeros(num_envs, dtype=np.int32)
        self.full = np.zeros(num_envs, dtype=bool)
        self.online_queue = [deque(maxlen=self.capacity // 10) for _ in range(num_envs)]

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        state: List[Dict[str, np.ndarray]],
    ):
        for i in range(self.num_envs):
            idx = self.idx[i]
            self.data["obs"][i, idx] = obs[i]
            self.data["action"][i, idx] = action[i]
            self.data["reward"][i, idx] = reward[i]
            self.data["done"][i, idx] = done[i]

            if state[i] is not None:
                state_to_store = {}
                for k, v in state[i].items():
                    if torch.is_tensor(v):
                        if v.dim() > 0 and v.shape[0] == 1:
                            state_to_store[k] = v.squeeze(0).cpu().numpy()
                        else:
                            state_to_store[k] = v.cpu().numpy()
                    else:
                        state_to_store[k] = v
                self.data["state"][i, idx] = state_to_store
            else:
                self.data["state"][i, idx] = None

            self.online_queue[i].append(
                {
                    "obs": obs[i],
                    "action": action[i],
                    "reward": reward[i],
                    "done": done[i],
                    "state": state_to_store if state[i] is not None else None,
                }
            )

            self.idx[i] = (self.idx[i] + 1) % self.capacity
            if self.idx[i] == 0:
                self.full[i] = True

    def update_state(self, env_idx: int, idx: int, state: Dict[str, np.ndarray]):
        state_to_store = {}
        for k, v in state.items():
            if torch.is_tensor(v):
                state_to_store[k] = v.cpu().numpy()
            else:
                state_to_store[k] = v
        self.data["state"][env_idx, idx] = state_to_store

    def sample(
        self, batch_size: int, seq_len: int, online_fraction: float = 0.5
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, int]]]:
        online_batch_size = int(batch_size * online_fraction)
        replay_batch_size = batch_size - online_batch_size

        batch = {
            "obs": [],
            "action": [],
            "reward": [],
            "done": [],
            "state": [],
        }
        indices = []

        if online_batch_size > 0:
            valid_envs = [
                i for i in range(self.num_envs) if len(self.online_queue[i]) >= seq_len
            ]
            if valid_envs:
                for _ in range(min(online_batch_size, len(valid_envs))):
                    env_idx = random.choice(valid_envs)
                    online_data = list(self.online_queue[env_idx])
                    if len(online_data) >= seq_len:
                        start = len(online_data) - seq_len
                        for key in ["obs", "action", "reward", "done"]:
                            seq = [online_data[start + i][key] for i in range(seq_len)]
                            batch[key].append(np.array(seq))

                        state_seq = online_data[start]["state"]
                        batch["state"].append(state_seq)
                        indices.append((-1, -1))

        for _ in range(replay_batch_size):
            env_idx = random.randint(0, self.num_envs - 1)
            max_idx = self.capacity if self.full[env_idx] else self.idx[env_idx]

            if max_idx < seq_len:
                continue

            valid_start = False
            attempts = 0
            while not valid_start and attempts < 100:
                start = random.randint(0, max_idx - seq_len)
                valid_start = True
                for i in range(seq_len - 1):
                    if self.data["done"][env_idx, (start + i) % self.capacity]:
                        valid_start = False
                        break
                attempts += 1

            if valid_start:
                for key in ["obs", "action", "reward", "done"]:
                    seq = []
                    for i in range(seq_len):
                        seq.append(self.data[key][env_idx, (start + i) % self.capacity])
                    batch[key].append(np.array(seq))

                batch["state"].append(self.data["state"][env_idx, start])
                indices.append((env_idx, start))

        for key in ["obs", "action", "reward", "done"]:
            if batch[key]:
                batch[key] = torch.FloatTensor(np.stack(batch[key]))
            else:
                batch[key] = torch.zeros(0, seq_len)

        return batch, indices

    def __len__(self):
        return np.sum(self.capacity * self.full + self.idx * ~self.full)


class DreamerV3:
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        config: Config,
        discrete: bool = False,
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.discrete = discrete
        self.obs_shape = obs_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_low = action_low
        self.action_high = action_high

        self.world_model = WorldModel(obs_shape, action_dim, config).to(self.device)
        self.actor = Actor(
            config.deter_size + config.stoch_size * config.stoch_discrete,
            action_dim,
            config,
            discrete,
            (
                torch.tensor(action_low, device=self.device)
                if action_low is not None
                else None
            ),
            (
                torch.tensor(action_high, device=self.device)
                if action_high is not None
                else None
            ),
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

        self.world_opt = LaProp(
            self.world_model.parameters(),
            lr=config.learning_rate,
            eps=config.laprop_eps,
        )
        self.actor_opt = LaProp(
            self.actor.parameters(),
            lr=config.learning_rate,
            eps=config.laprop_eps,
        )
        self.critic_opt = LaProp(
            self.critic.parameters(),
            lr=config.learning_rate,
            eps=config.laprop_eps,
        )

        self.replay_buffer = ParallelReplayBuffer(config.buffer_size, config.num_envs)

        self.return_normalizer = ReturnNormalizer(
            decay=config.return_norm_decay,
            limit=config.return_norm_limit,
            device=self.device,
        )

        self.train_steps = 0
        self.prev_actions = [None] * config.num_envs
        self.prev_states = [None] * config.num_envs

    def process_observation(self, obs: np.ndarray) -> torch.Tensor:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)

        if len(self.obs_shape) > 1 and len(obs_t.shape) == len(self.obs_shape):
            obs_t = obs_t.reshape(-1, *self.obs_shape)

        return obs_t

    def act(
        self,
        obs: np.ndarray,
        states: Optional[List[Dict[str, torch.Tensor]]] = None,
        training: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, torch.Tensor]]]:
        with torch.no_grad():
            num_envs = obs.shape[0]
            obs_t = self.process_observation(obs)

            if states is None:
                states = [None] * num_envs

            batch_states = []
            for i in range(num_envs):
                if states[i] is None:
                    initial = self.world_model.rssm.initial_state(1, self.device)
                    batch_states.append({k: v for k, v in initial.items()})
                else:
                    state_dict = {}
                    for k, v in states[i].items():
                        if v.dim() == 1:
                            state_dict[k] = v.unsqueeze(0)
                        else:
                            state_dict[k] = v
                    batch_states.append(state_dict)

            batch_deter = torch.cat([s["deter"] for s in batch_states], dim=0)
            batch_stoch = torch.cat([s["stoch"] for s in batch_states], dim=0)
            batch_state = {"deter": batch_deter, "stoch": batch_stoch}

            embed = self.world_model.rssm.encoder(obs_t)

            prev_actions = []
            for i in range(num_envs):
                if self.prev_actions[i] is None:
                    if self.discrete:
                        prev_action = torch.zeros(1, self.actor.head.out_features).to(
                            self.device
                        )
                    else:
                        action_dim = self.actor.head.out_features // 2
                        prev_action = torch.zeros(1, action_dim).to(self.device)
                else:
                    prev_action = (
                        torch.FloatTensor(self.prev_actions[i])
                        .unsqueeze(0)
                        .to(self.device)
                    )
                prev_actions.append(prev_action.squeeze(0))

            prev_actions = torch.stack(prev_actions)

            prev_stoch = batch_state["stoch"].view(num_envs, -1)
            deter = self.world_model.rssm.gru(
                torch.cat([prev_stoch, prev_actions], dim=-1), batch_state["deter"]
            )

            posterior_logits = self.world_model.rssm.posterior_net(
                torch.cat([deter, embed], dim=-1)
            )
            posterior_logits = posterior_logits.view(
                -1, self.config.stoch_size, self.config.stoch_discrete
            )
            posterior = OneHotDist(logits=posterior_logits, unimix=self.config.unimix)
            stoch = posterior.mode if not training else posterior.rsample()

            new_states = {"deter": deter, "stoch": stoch}

            feat = self.world_model.decoder.get_feat(new_states)

            if training:
                if self.discrete:
                    action_t, _, _ = self.actor(feat, training=True)
                else:
                    action_t, _, _ = self.actor(feat, training=True)
            else:
                action_t = self.actor(feat, training=False)

            actions_np = action_t.cpu().numpy()

            for i in range(num_envs):
                if self.discrete:
                    action = actions_np[i]
                    action_onehot = np.zeros(self.actor.head.out_features)
                    action_onehot[action] = 1.0
                    self.prev_actions[i] = action_onehot
                    actions_np[i] = action
                else:
                    self.prev_actions[i] = actions_np[i]

                self.prev_states[i] = {
                    "deter": deter[i].detach(),
                    "stoch": stoch[i].detach(),
                }

            output_states = []
            for i in range(num_envs):
                output_states.append(
                    {"deter": deter[i : i + 1], "stoch": stoch[i : i + 1]}
                )

            return actions_np, output_states

    def train(self, steps: int = 1):
        if (
            len(self.replay_buffer)
            < self.config.batch_size * self.config.sequence_length
        ):
            return {}

        all_metrics = {}

        for _ in range(steps):
            batch, indices = self.replay_buffer.sample(
                self.config.batch_size,
                self.config.sequence_length,
                self.config.online_fraction,
            )

            if batch["obs"].shape[0] == 0:
                continue

            for k, v in batch.items():
                if k != "state":
                    batch[k] = v.to(self.device)

            initial_states = []
            for state_dict in batch["state"]:
                if state_dict is not None:
                    initial_state = {
                        "deter": torch.tensor(state_dict["deter"], device=self.device),
                        "stoch": torch.tensor(state_dict["stoch"], device=self.device),
                    }
                else:
                    initial_state = self.world_model.rssm.initial_state(1, self.device)
                    initial_state = {k: v.squeeze(0) for k, v in initial_state.items()}
                initial_states.append(initial_state)

            initial_state_batch = {
                "deter": torch.stack([s["deter"] for s in initial_states]),
                "stoch": torch.stack([s["stoch"] for s in initial_states]),
            }

            world_metrics = self._train_world_model(batch, initial_state_batch, indices)
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
                    target_param.data.mul_(self.config.ema_decay).add_(
                        param.data, alpha=(1 - self.config.ema_decay)
                    )

            self.train_steps += 1

        for k in all_metrics:
            all_metrics[k] /= steps

        return all_metrics

    def _train_world_model(
        self,
        batch: Dict[str, torch.Tensor],
        initial_state: Dict[str, torch.Tensor],
        indices: List[Tuple[int, int]],
    ) -> Dict[str, float]:
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        done = batch["done"]

        (
            states,
            obs_pred,
            reward_logits,
            continue_logits,
            kl_dyns,
            kl_reps,
            kl_dyns_raw,
            kl_reps_raw,
        ) = self.world_model(obs, action, initial_state)

        obs_flat = obs.reshape(obs.shape[0], obs.shape[1], -1)

        if self.world_model.rssm.is_image:
            obs_normalized = obs_flat / 255.0 if obs_flat.max() > 1.0 else obs_flat
            obs_loss = F.mse_loss(obs_pred, obs_normalized, reduction="mean")
        else:
            obs_loss = 0.5 * (symlog(obs_pred) - symlog(obs_flat)).pow(2).mean()

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

        dynamics_loss = self.config.dyn_weight * kl_dyns.mean()
        representation_loss = self.config.rep_weight * kl_reps.mean()
        prediction_loss = self.config.pred_weight * (
            obs_loss + reward_loss + continue_loss
        )

        loss = prediction_loss + dynamics_loss + representation_loss

        self.world_opt.zero_grad(set_to_none=True)
        loss.backward()
        adaptive_gradient_clip_model(
            self.world_model, self.config.agc_clip_factor, self.config.agc_eps
        )
        self.world_opt.step()

        with torch.no_grad():
            for i, (env_idx, start_idx) in enumerate(indices):
                if env_idx >= 0:
                    final_state = {
                        "deter": states["deter"][i, -1].cpu().numpy(),
                        "stoch": states["stoch"][i, -1].cpu().numpy(),
                    }
                    self.replay_buffer.update_state(
                        env_idx,
                        (start_idx + self.config.sequence_length - 1)
                        % self.replay_buffer.capacity,
                        final_state,
                    )

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

            if self.world_model.rssm.is_image:
                obs_normalized = obs_flat / 255.0 if obs_flat.max() > 1.0 else obs_flat
                obs_error = F.mse_loss(obs_pred, obs_normalized, reduction="mean")
            else:
                obs_error = 0.5 * (symlog(obs_pred) - symlog(obs_flat)).pow(2).mean()

        return {
            "world/total_loss": loss.item(),
            "world/prediction_loss": prediction_loss.item(),
            "world/dynamics_loss": dynamics_loss.item(),
            "world/representation_loss": representation_loss.item(),
            "world/obs_loss": obs_loss.item(),
            "world/reward_loss": reward_loss.item(),
            "world/continue_loss": continue_loss.item(),
            "world/kl_dyn": kl_dyns.mean().item(),
            "world/kl_rep": kl_reps.mean().item(),
            "world/kl_dyn_raw": kl_dyns_raw.mean().item(),
            "world/kl_rep_raw": kl_reps_raw.mean().item(),
            "world/reward_error": reward_error.item(),
            "world/continue_accuracy": continue_acc.item(),
            "world/obs_error": obs_error.item(),
        }

    def _train_actor_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            obs = batch["obs"][:, 0]
            obs = self.process_observation(obs.cpu().numpy())

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

            batch_size = state["deter"].shape[0]
            horizon = self.config.horizon

            all_rewards = torch.zeros(batch_size, horizon, device=self.device)
            all_continues = torch.zeros(batch_size, horizon, device=self.device)
            all_states = []

            for t in range(horizon):
                feat = self.world_model.decoder.get_feat(state)
                action, _, _ = self.actor(feat, training=True)
                if self.discrete:
                    action_onehot = F.one_hot(
                        action, num_classes=self.actor.head.out_features
                    ).float()
                    state = self.world_model.rssm.imagine(action_onehot, state)
                else:
                    state = self.world_model.rssm.imagine(action, state)
                all_states.append({k: v.clone() for k, v in state.items()})

                _, reward_logits, continue_logits = self.world_model.decoder(state)

                reward_probs = F.softmax(reward_logits, dim=-1)
                reward = twohot_decode(
                    reward_probs,
                    self.config.twohot_bins,
                    self.config.twohot_min,
                    self.config.twohot_max,
                )
                all_rewards[:, t] = reward.squeeze()

                cont = torch.sigmoid(continue_logits).squeeze(-1)
                all_continues[:, t] = cont

            final_feat = self.world_model.decoder.get_feat(state)
            final_value_logits = self.target_critic(final_feat)
            final_value_probs = F.softmax(final_value_logits, dim=-1)
            final_value = twohot_decode(
                final_value_probs,
                self.config.twohot_bins,
                self.config.twohot_min,
                self.config.twohot_max,
            )

            all_values = []
            for state in all_states:
                feat = self.world_model.decoder.get_feat(state)
                value_logits = self.target_critic(feat)
                value_probs = F.softmax(value_logits, dim=-1)
                value = twohot_decode(
                    value_probs,
                    self.config.twohot_bins,
                    self.config.twohot_min,
                    self.config.twohot_max,
                )
                all_values.append(value)

            values = torch.stack(all_values, dim=1)
            bootstrap_values = torch.cat(
                [values[:, 1:], final_value.unsqueeze(1)], dim=1
            )
            returns = self._compute_lambda_returns(
                all_rewards, bootstrap_values, all_continues
            )

            self.return_normalizer.update(returns.flatten())

        imagination_loss = 0

        for t in range(self.config.horizon):
            feat = self.world_model.decoder.get_feat(all_states[t])
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
            imagination_loss = imagination_loss + loss.mean()

        imagination_loss = imagination_loss / self.config.horizon

        replay_loss = 0

        replay_states, _, _, _, _, _, _, _ = self.world_model(
            batch["obs"], batch["action"]
        )

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
                    action, _, _ = self.actor(feat, training=True)

                    # Convert discrete action indices to one-hot for RSSM
                    if self.discrete:
                        action_onehot = F.one_hot(
                            action, num_classes=self.actor.head.out_features
                        ).float()
                        state = self.world_model.rssm.imagine(action_onehot, state)
                    else:
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
            replay_loss = replay_loss + loss.mean()

        if num_replay_batches > 0:
            replay_loss = replay_loss / num_replay_batches

        critic_loss = (
            self.config.critic_weight * imagination_loss
            + self.config.critic_replay_weight * replay_loss
        )

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        adaptive_gradient_clip_model(
            self.critic, self.config.agc_clip_factor, self.config.agc_eps
        )
        self.critic_opt.step()

        with torch.no_grad():
            if len(all_states) > 0:
                sample_feat = self.world_model.decoder.get_feat(all_states[0])
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
        log_probs = []
        entropies = []

        state = initial_state
        for _ in range(self.config.horizon):
            states.append(state)

            feat = self.world_model.decoder.get_feat(state)
            action, log_prob, entropy = self.actor(feat, training=True)
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)

            # Convert discrete action indices to one-hot for RSSM
            if self.discrete:
                action_onehot = F.one_hot(
                    action, num_classes=self.actor.head.out_features
                ).float()
                state = self.world_model.rssm.imagine(action_onehot, state)
            else:
                state = self.world_model.rssm.imagine(action, state)

        with torch.no_grad():
            batch_size = initial_state["deter"].shape[0]
            horizon = self.config.horizon

            all_rewards = torch.zeros(batch_size, horizon, device=self.device)
            all_continues = torch.zeros(batch_size, horizon, device=self.device)
            all_values = torch.zeros(batch_size, horizon, device=self.device)

            for t, state in enumerate(states):
                _, reward_logits, continue_logits = self.world_model.decoder(state)

                reward_probs = F.softmax(reward_logits, dim=-1)
                reward = twohot_decode(
                    reward_probs,
                    self.config.twohot_bins,
                    self.config.twohot_min,
                    self.config.twohot_max,
                )
                all_rewards[:, t] = reward.squeeze()

                cont = torch.sigmoid(continue_logits).squeeze(-1)
                all_continues[:, t] = cont

                feat = self.world_model.decoder.get_feat(state)
                value_logits = self.critic(feat)
                value_probs = F.softmax(value_logits, dim=-1)
                value = twohot_decode(
                    value_probs,
                    self.config.twohot_bins,
                    self.config.twohot_min,
                    self.config.twohot_max,
                )
                all_values[:, t] = value.squeeze()

            final_feat = self.world_model.decoder.get_feat(state)
            final_value_logits = self.target_critic(final_feat)
            final_value_probs = F.softmax(final_value_logits, dim=-1)
            final_value = twohot_decode(
                final_value_probs,
                self.config.twohot_bins,
                self.config.twohot_min,
                self.config.twohot_max,
            )

            bootstrap_values = torch.cat(
                [all_values[:, 1:], final_value.unsqueeze(1)], dim=1
            )
            returns = self._compute_lambda_returns(
                all_rewards, bootstrap_values, all_continues
            )

            scale = max(self.config.return_norm_limit, self.return_normalizer.scale)
            advantages = (returns - all_values) / scale

        total_pg_loss = 0
        total_entropy = 0

        for t in range(self.config.horizon):
            pg_loss = -(advantages[:, t].detach() * log_probs[t]).mean()
            total_pg_loss = total_pg_loss + pg_loss
            total_entropy = total_entropy + entropies[t].mean()

        total_pg_loss = total_pg_loss / self.config.horizon
        total_entropy = total_entropy / self.config.horizon

        total_loss = total_pg_loss - self.config.entropy_scale * total_entropy

        self.actor_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        adaptive_gradient_clip_model(
            self.actor, self.config.agc_clip_factor, self.config.agc_eps
        )
        self.actor_opt.step()

        return {
            "actor/loss": total_loss.item(),
            "actor/pg_loss": total_pg_loss.item(),
            "actor/entropy": total_entropy.item(),
            "actor/mean_return": returns.mean().item(),
            "actor/mean_return_normalized": (returns / scale).mean().item(),
            "actor/mean_value": all_values.mean().item(),
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


def get_action_repeat(env_name: str) -> int:
    if "Atari" in env_name or "ALE" in env_name:
        return 4
    else:
        return 2


def make_env(env_name: str):
    def _init():
        env = gym.make(env_name, render_mode=None)
        return env

    return _init


def train_dreamer(
    env_name: str, total_steps: int = 1000000
) -> Tuple[DreamerV3, List[float]]:
    config = Config()
    config.action_repeat = get_action_repeat(env_name)

    env_fns = [make_env(env_name) for _ in range(config.num_envs)]
    envs = AsyncVectorEnv(env_fns)

    single_env = gym.make(env_name)
    obs_shape = single_env.observation_space.shape
    is_image = len(obs_shape) > 1

    if isinstance(single_env.action_space, gym.spaces.Discrete):
        action_dim = single_env.action_space.n
        discrete = True
        action_low = None
        action_high = None
    else:
        action_dim = single_env.action_space.shape[0]
        discrete = False
        action_low = single_env.action_space.low
        action_high = single_env.action_space.high

    single_env.close()

    agent = DreamerV3(obs_shape, action_dim, config, discrete, action_low, action_high)

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
    print(f"  Environment: {env_name_colored} x{config.num_envs}")
    print(f"  Observation shape: {obs_shape}, Action dim: {action_dim} ({action_type})")
    print(f"  Using {encoder_type} encoder")
    print(f"  Action repeat: {config.action_repeat}")
    print(
        f"  Model: deter={config.deter_size}, stoch={config.stoch_size}×{config.stoch_discrete}, hidden={config.hidden_size}"
    )
    print(f"  Learning rates: {config.learning_rate}")
    print(f"  Replay ratio: {config.replay_ratio}\n")

    episode_rewards = [[] for _ in range(config.num_envs)]
    episode_count = 0
    total_env_steps = 0
    update_count = 0
    best_reward = -float("inf")

    warmup_steps = config.batch_size * config.sequence_length

    spinner = spinning_cursor()
    start_time = time.time()

    os.makedirs("models", exist_ok=True)

    obs = envs.reset()[0]
    states = None
    episode_reward = np.zeros(config.num_envs)

    for i in range(config.num_envs):
        agent.prev_actions[i] = None

    while total_env_steps < total_steps:
        actions, states = agent.act(obs, states, training=True)

        cum_rewards = np.zeros(config.num_envs)
        dones = np.zeros(config.num_envs, dtype=bool)

        for _ in range(config.action_repeat):
            next_obs, rewards, terminateds, truncateds, _ = envs.step(actions)
            cum_rewards += rewards
            dones |= terminateds | truncateds
            if dones.all():
                break

        action_data = []
        state_data = []
        for i in range(config.num_envs):
            if discrete:
                action_onehot = np.zeros(action_dim)
                action_onehot[actions[i]] = 1
                action_data.append(action_onehot)
            else:
                action_data.append(actions[i])
            state_data.append(agent.prev_states[i])

        agent.replay_buffer.add(
            obs, np.array(action_data), cum_rewards, dones.astype(float), state_data
        )

        episode_reward += cum_rewards
        obs = next_obs
        total_env_steps += config.num_envs * config.action_repeat

        if (
            total_env_steps > warmup_steps
            and total_env_steps % config.replay_ratio == 0
        ):
            metrics = agent.train()
            update_count += 1

            for key, value in metrics.items():
                writer.add_scalar(f"train/{key}", value, total_env_steps)

            if update_count % 100 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = total_env_steps / elapsed
                eta = (
                    (total_steps - total_env_steps) / steps_per_sec
                    if steps_per_sec > 0
                    else 0
                )

                separator = colored("━" * 95, Colors.DIM)
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
                world_kl_dyn = metrics.get("world/kl_dyn", 0)
                kl_dyn_raw = metrics.get("world/kl_dyn_raw", 0)
                world_kl_rep = metrics.get("world/kl_rep", 0)
                kl_rep_raw = metrics.get("world/kl_rep_raw", 0)
                critic_loss = metrics.get("critic/loss", 0)
                critic_scale = metrics.get("critic/return_scale", 0)
                actor_loss = metrics.get("actor/loss", 0)
                actor_entropy = metrics.get("actor/entropy", 0)
                actor_adv = metrics.get("actor/mean_advantage", 0)
                reward_error = metrics.get("world/reward_error", 0)
                obs_error = metrics.get("world/obs_error", 0)

                losses_header = colored("🧠 Losses:", Colors.BLUE)
                world_loss_str = colored(f"{world_loss:.4f}", Colors.CYAN)
                world_kl_dyn_str = colored(f"{world_kl_dyn:.4f}", Colors.CYAN)
                world_kl_rep_str = colored(f"{world_kl_rep:.4f}", Colors.CYAN)
                raw_kl_dyn_str = colored(f"{kl_dyn_raw:.4f}", Colors.CYAN)
                raw_kl_rep_str = colored(f"{kl_rep_raw:.4f}", Colors.CYAN)
                critic_loss_str = colored(f"{critic_loss:.4f}", Colors.CYAN)
                critic_scale_str = colored(f"{critic_scale:.4f}", Colors.CYAN)
                actor_loss_str = colored(f"{actor_loss:.4f}", Colors.CYAN)
                actor_entropy_str = colored(f"{actor_entropy:.4f}", Colors.CYAN)
                actor_adv_str = colored(f"{actor_adv:.4f}", Colors.CYAN)
                spacing_str = "           "

                pred_header = colored("🔮 Prediction:", Colors.BLUE)
                reward_error_str = colored(f"{reward_error:.4f}", Colors.CYAN)
                obs_error_str = colored(f"{obs_error:.4f}", Colors.CYAN)

                print(
                    f"{losses_header} "
                    f"World: {world_loss_str}, "
                    f"KL(dyn): {world_kl_dyn_str} (raw: {raw_kl_dyn_str}), "
                    f"KL(rep): {world_kl_rep_str} (raw: {raw_kl_rep_str})\n"
                    f"{spacing_str}Critic: {critic_loss_str} (scale: {critic_scale_str}), "
                    f"Actor: {actor_loss_str} (advantage: {actor_adv_str}), "
                    f"Entropy: {actor_entropy_str}"
                )
                print(
                    f"{pred_header} "
                    f"Reward error: {reward_error_str}, "
                    f"Obs error: {obs_error_str}"
                )

                all_episode_rewards = [
                    r for env_rewards in episode_rewards for r in env_rewards
                ]
                if all_episode_rewards:
                    recent_avg = np.mean(all_episode_rewards[-50:])
                    last_reward = all_episode_rewards[-1]
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

        for i in range(config.num_envs):
            if dones[i]:
                episode_rewards[i].append(episode_reward[i])
                episode_count += 1

                writer.add_scalar("episode/reward", episode_reward[i], episode_count)

                if total_env_steps > warmup_steps and episode_reward[i] > best_reward:
                    best_reward = episode_reward[i]
                    best_model_path = f"models/dreamer_v3_{env_name_safe}_best.pt"
                    torch.save(
                        {
                            "world_model": agent.world_model.state_dict(),
                            "actor": agent.actor.state_dict(),
                            "critic": agent.critic.state_dict(),
                        },
                        best_model_path,
                    )
                    save_msg = colored("💾 New best model saved!", Colors.GREEN)
                    reward_str = colored(f"{episode_reward[i]:.2f}", Colors.BOLD)
                    print(f"\n{save_msg} Reward: {reward_str}")

                episode_reward[i] = 0
                agent.prev_actions[i] = None
                if states is not None:
                    states[i] = None

        if total_env_steps <= warmup_steps:
            warmup_label = colored("🔥 Warmup", Colors.YELLOW)
            all_rewards = [r for env_rewards in episode_rewards for r in env_rewards]
            avg_reward = np.mean(all_rewards) if all_rewards else 0
            reward_str = colored(f"{avg_reward:.2f}", Colors.CYAN)
            print(
                f"\r{warmup_label} {next(spinner)} "
                + f"Step {total_env_steps}/{warmup_steps}, "
                + f"Episodes {episode_count}, "
                + f"Avg Reward: {reward_str}",
                end=" ",
                flush=True,
            )

    envs.close()
    writer.close()

    complete_msg = colored("✅ Training Complete!", Colors.HEADER)
    time_elapsed = (time.time() - start_time) / 60
    time_str = colored(f"{time_elapsed:.1f} minutes", Colors.CYAN)

    print(f"\n{complete_msg}")
    print(f"Total time: {time_str}")

    all_rewards = [r for env_rewards in episode_rewards for r in env_rewards]
    return agent, all_rewards


def evaluate_and_save_gif(agent: DreamerV3, env_name: str, num_episodes: int = 5):
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

        agent.prev_actions[0] = None

        while not done:
            frames.append(env.render())

            action, state_list = agent.act(
                np.expand_dims(obs, 0), [state] if state else None, training=False
            )
            action = action[0]
            state = state_list[0]

            cum_reward = 0
            for _ in range(agent.config.action_repeat):
                obs, reward, terminated, truncated, _ = env.step(action)
                cum_reward += reward
                done = terminated or truncated
                if done:
                    break

            episode_reward += cum_reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        reward_str = colored(f"{episode_reward:.2f}", Colors.CYAN)
        print(f"  Episode {ep+1}: Reward = {reward_str}, Length = {episode_length}")

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_frames = frames

    env.close()

    summary_header = colored("📊 Evaluation Summary:", Colors.HEADER)
    best_str = colored(f"{best_reward:.2f}", Colors.GREEN)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_str = colored(f"{mean_reward:.2f}", Colors.GREEN)
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
        gif_msg = colored(gif_filename, Colors.HEADER)
        print(f"  Saved GIF: {gif_msg}")

    return episode_rewards


def main():
    environments = [
        # "CartPole-v1",
        # "Pendulum-v1",
        "LunarLander-v3",
        "Ant-v5",
        "CarRacing-v3",
        "BipedalWalker-v3",
        "ALE/Pong-v5",
    ]

    results = {}
    separator = "=" * 95

    for env_name in environments:
        try:
            header_msg = colored(f"🚀 Training Dreamer-v3 on {env_name}", Colors.HEADER)

            print(f"\n{separator}")
            print(f"{header_msg}")
            print(f"{separator}")

            total_steps = 2_000_000

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
                load_msg = colored("📂 Loaded best model for evaluation", Colors.HEADER)
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
            traceback.print_exc()
            exit()

    final_header = colored("✅ TRAINING COMPLETE - FINAL SUMMARY", Colors.HEADER)

    print(f"\n{separator}")
    print(f"{final_header}")
    print(f"{separator}")

    for env_name, res in results.items():
        env_colored = colored(env_name, Colors.MAGENTA)
        train_time_mins = res["train_time"] / 60
        final_avg = res["final_avg_reward"]
        final_avg_str = colored(f"{final_avg:.2f}", Colors.GREEN)
        eval_mean = np.mean(res["eval_rewards"])
        eval_std = np.std(res["eval_rewards"])
        eval_mean_str = colored(f"{eval_mean:.2f}", Colors.GREEN)
        num_episodes = len(res["train_rewards"])

        print(f"\n{env_colored}:")
        print(f"  Training Time: {train_time_mins:.1f} minutes")
        print(f"  Final Training Avg: {final_avg_str}")
        print(f"  Eval Mean: {eval_mean_str} ± {eval_std:.2f}")
        print(f"  Episodes: {num_episodes}")


if __name__ == "__main__":
    main()
