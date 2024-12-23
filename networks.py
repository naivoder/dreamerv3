import torch
import torch.nn.functional as F
from torch import distributions as td
import numpy as np
import math
import re

import utils


class RSSM(torch.nn.Module):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        normalize=True,
        activation="SiLU",
        action_mean="none",
        action_std="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
    ):
        super(RSSM, self).__init__()
        self.stoch = stoch
        self.deter = deter
        self.hidden = hidden
        self.rec_depth = rec_depth
        self.discrete = discrete
        self.activation = getattr(torch.nn, activation)
        self.action_mean = action_mean
        self.action_std = action_std
        self.min_std = min_std
        self.unimix_ratio = unimix_ratio
        self.initial = initial
        self.num_actions = num_actions
        self.embed = embed
        self.device = device

        input_layers = list()
        if self.discrete:
            input_shape = self.stoch * self.discrete + num_actions
        else:
            input_shape = self.stoch + num_actions
        input_layers.append(torch.nn.Linear(input_shape, hidden), bias=False)
        if normalize:
            input_layers.append(torch.nn.LayerNorm(hidden, eps=1e-3))
        input_layers.append(activation())
        self.input_layers = torch.nn.Sequential(*input_layers)
        self.input_layers.apply(utils.weight_init)
        self.rnn = torch.nn.GRUCell(hidden, self.deter, norm=normalize)
        self.rnn.apply(utils.weight_init)

        img_output_layers = list()
        input_shape = self.deter
        img_output_layers.append(torch.nn.Linear(input_shape, hidden), bias=False)
        if normalize:
            img_output_layers.append(torch.nn.LayerNorm(hidden, eps=1e-3))
        img_output_layers.append(activation())
        self.img_output_layers = torch.nn.Sequential(*img_output_layers)
        self.img_output_layers.apply(utils.weight_init)

        obs_output_layers = list()
        input_shape = self.deter + self.embed
        obs_output_layers.append(torch.nn.Linear(input_shape, hidden), bias=False)
        if normalize:
            obs_output_layers.append(torch.nn.LayerNorm(hidden, eps=1e-3))
        obs_output_layers.append(activation())
        self.obs_output_layers = torch.nn.Sequential(*obs_output_layers)
        self.obs_output_layers.apply(utils.weight_init)

        if self.discrete:
            self.img_stat_layer = torch.nn.Linear(hidden, self.stoch * self.discrete)
            self.img_stat_layer.apply(utils.uniform_weight_init(1.0))
            self.obs_stat_layer = torch.nn.Linear(hidden, self.stoch * self.discrete)
            self.obs_stat_layer.apply(utils.uniform_weight_init(1.0))
        else:
            self.img_stat_layer = torch.nn.Linear(hidden, self.stoch * 2)
            self.img_stat_layer.apply(utils.uniform_weight_init(1.0))
            self.obs_stat_layer = torch.nn.Linear(hidden, self.stoch * 2)
            self.obs_stat_layer.apply(utils.uniform_weight_init(1.0))

        if self.initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self.deter), device=torch.device(self.device)),
                requires_grad=True,
            )

    def init_state(self, batch_size):
        deter = torch.zeros(batch_size, self.deter, device=self.device)
        if self.discrete:
            shape = (batch_size, self.stoch, self.discrete)
            state = dict(
                logit=torch.zeros(shape, device=self.device),
                stoch=torch.zeros(shape, device=self.device),
                deter=deter,
            )
        else:
            shape = (batch_size, self.stoch)
            state = dict(
                mean=torch.zeros(shape, device=self.device),
                std=torch.zeros(shape, device=self.device),
                stoch=torch.zeros(shape, device=self.device),
                deter=deter,
            )
        if self.initial == "zeros":
            return state
        elif self.initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self.initial)

    def observe(self, embed, action, is_first, state=None):
        """
        Process a sequence of observations and actions to generate posterior and prior latent states.
        """
        # (B, T, C) -> (T, B, C)
        swap = lambda x: torch.moveaxis(x, 0, 1)
        embed, action, is_first = map(swap, (embed, action, is_first))

        # prev_state[0] == passing only the posterior from the previous step
        posterior, prior = utils.static_scan(
            lambda prev_state, prev_action, embed, is_first: self.obs_step(
                prev_state[0], prev_action, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (T, B, stoch, discrete) -> (B, T, stoch, discrete)
        posterior = {k: swap(v) for k, v in posterior.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return posterior, prior

    def imagine(self, state, action):
        """
        Perform a sequence rollout using the prior model, given a sequence of actions.
        """
        action = torch.moveaxis(action, 0, 1)
        prior = utils.static_scan(self.img_step, [action], state)
        prior = {k: torch.moveaxis(v, 0, 1) for k, v in prior.items()}
        return prior

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """
        Perform one step of observation update, combining prior and observed embeddings.
        """
        # Initialize state if the previous state is None or if this is the first step
        if prev_state is None or torch.sum(is_first) == len(is_first):
            prev_state = self.init_state(len(is_first))
            prev_action = torch.zeros(
                (len(is_first), self.num_actions), device=self.device
            )
        # Reset state for episodes where is_first is True
        elif torch.sum(is_first) > 0:
            is_first = is_first.unsqueeze(-1)
            prev_action *= 1.0 - is_first  # zero-out actions for new episodes
            init_state = self.init_state(len(is_first))
            for k, v in prev_state.items():
                _is_first = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(v.shape) - len(is_first.shape)),
                )
                prev_state[k] = v * (1.0 - _is_first) + init_state[k] * _is_first

        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], dim=-1)
        x = self.obs_output_layers(x)
        stats = self._suff_stats_layer("obs", x)

        stoch = self.get_dist(stats).sample() if sample else self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        """
        Perform one step of image (prior) prediction.
        """
        prev_stoch = prev_state["stoch"]
        if self.discrete:
            shape = list(prev_stoch.shape[:-2]) + [self.stoch * self.discrete]
            prev_stoch = prev_stoch.reshape(shape)

        x = torch.cat([prev_stoch, prev_action], dim=-1)
        x = self.img_input_layers(x)

        for _ in range(self.rec_depth):
            deter = prev_state["deter"]
            x, deter = self.rnn(x, deter)
            deter = deter[0]

        x = self.img_output_layers(x)
        stats = self._suff_stats_layer("img", x)

        stoch = self.get_dist(stats).sample() if sample else self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_feats(self, state):
        stoch = state["stoch"]
        if self.discrete:
            shape = list(stoch.shape[:-2]) + [self.stoch * self.discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], dim=-1)

    def get_dist(self, state):
        if self.discrete:
            logits = state["logits"]
            dist = td.Independent(
                utils.OneHotDist(logits, unimix_ratio=self.unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = utils.ContinuousDist(td.Independent(td.Normal(mean, std), 1))
        return dist

    def get_stoch(self, deter):
        x = self.img_output_layers(deter)
        stats = self._suff_stats_layer("img", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if name == "img":
            x = self.img_stat_layer(x)
        elif name == "obs":
            x = self.obs_stat_layer(x)
        else:
            raise NotImplementedError(name)

        if self.discrete:
            logits = x.reshape(list(x.shape[:-1]) + [self.stoch, self.discrete])
            return {"logits": logits}
        else:
            mean, std = torch.split(x, [self.stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self.action_mean]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self.action_std]()
            std = std + self.min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        dist = lambda x: self.get_dist(x)
        stop_grad = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = td.kl.kl_divergence(
            dist(post) if self.discrete else dist(post)._dist,
            dist(stop_grad(prior)) if self.discrete else dist(stop_grad(prior))._dist,
        )
        dyn_loss = td.kl.kl_divergence(
            (dist(stop_grad(post)) if self.discrete else dist(stop_grad(post))._dist),
            dist(prior) if self.discrete else dist(prior)._dist,
        )

        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss
        return loss, value, dyn_loss, rep_loss


class MLPEncoder(torch.nn.Module):
    def __init__(self):
        pass


class MLPDecoder(torch.nn.Module):
    def __init__(self):
        pass


class GRUCell(torch.nn.Module):
    def __init__(self):
        pass


class ConvEncoder(torch.nn.Module):
    def __init__(self):
        pass


class ConvDecoder(torch.nn.Module):
    def __init__(self):
        pass


class MLP(torch.nn.Module):
    def __init__(self):
        pass


class Conv2dSamePad(torch.nn.Conv2d):
    pass


class ImgChLayerNorm(torch.nn.Module):
    def __init__(self):
        pass
