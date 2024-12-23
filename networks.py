import torch
import torch.nn.functional as F
from torch import distributions as td
import numpy as np
import math
import re


class RSSM(torch.nn.Module):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
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
