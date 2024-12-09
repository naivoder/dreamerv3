import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from utils import gumbel_softmax, symlog, symexp, ObsNormalizer
from networks import WorldModel, Actor, Critic
from memory import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DreamerV3:
    def __init__(self, obs_shape, act_dim, is_image, is_discrete, config):
        self.config = config
        self.obs_shape = obs_shape
        self.is_image = is_image
        self.is_discrete = is_discrete
        self.env_name = config.env
        self.act_dim = act_dim

        pass

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = 1.0 / denoms
            std = np.sqrt(scale) / 0.87962566103423978
            nn.init.trunc_normal_(
                m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
            )
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
            denoms = (in_num + out_num) / 2.0
            scale = 1.0 / denoms
            std = np.sqrt(scale) / 0.87962566103423978
            nn.init.trunc_normal_(
                m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
            )
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    def reset_hidden_states(self):
        pass

    def update_world_model(self):
        pass

    def update_actor(self):
        pass

    def update_critic(self):
        pass

    def lambda_return(self, reward, value, pcont, bootstrap, lambda_):
        next_values = torch.cat([value[1:], bootstrap[None]], 0)
        inputs = reward + pcont * next_values * (1 - lambda_)
        returns = self._static_scan(
            lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
            (inputs, pcont),
            bootstrap,
        )
        return returns

    def _static_scan(self, fn, inputs, start):
        last = start
        outputs = []
        for index in reversed(range(inputs[0].shape[0])):
            inp = [input[index] for input in inputs]
            last = fn(last, *inp)
            outputs.append(last)
        outputs = torch.stack(outputs[::-1], dim=0)
        return outputs

    def _soft_update(self, target, source, tau=None):
        if tau is None:
            tau = self.config.tau
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def policy(self, obs, env_id, reset=False):
        pass

    def remember(self, env_id, obs, act, rew, next_obs, done):
        pass

    def train(self, num_updates):
        pass

    def save_checkpoint(self):
        torch.save(
            self.world_model.state_dict(), f"weights/{self.env_name}_world_model.pt"
        )
        torch.save(self.actor.state_dict(), f"weights/{self.env_name}_actor.pt")
        torch.save(self.critic.state_dict(), f"weights/{self.env_name}_critic.pt")

    def load_checkpoint(self):
        self.world_model.load_state_dict(
            torch.load(f"weights/{self.env_name}_world_model.pt")
        )
        self.actor.load_state_dict(torch.load(f"weights/{self.env_name}_actor.pt"))
        self.critic.load_state_dict(torch.load(f"weights/{self.env_name}_critic.pt"))
