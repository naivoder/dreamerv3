import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.transforms import TanhTransform
from torch.distributions import (
    Normal,
    Independent,
    TransformedDistribution,
    Categorical,
    Bernoulli,
)


class ObservationEncoder(nn.Module):
    def __init__(self, input_shape, depth, kernels):
        super(ObservationEncoder, self).__init__()
        self.depth = depth
        self.kernels = kernels
        self.input_shape = input_shape

        conv_layers = []
        c_in = self.input_shape[0]
        for i, kernel in enumerate(self.kernels):
            c_out = (2**i) * self.depth
            conv = nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=2, padding=0)
            conv_layers.append(conv)
            c_in = c_out
        self.convs = nn.ModuleList(conv_layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # observation: [B,T,H,W,C]
        B, T, C, H, W = observation.shape
        x = x.view(B * T, C, H, W)

        for conv in self.convs:
            x = F.relu(conv(x))

        x = x.view(B, T, -1)
        return x


class ObservationDecoder(nn.Module):
    def __init__(
        self,
        depth,
        kernels,
        output_shape,
    ):
        super(ObservationDecoder, self).__init__()
        self.depth = depth
        self.kernels = kernels
        self.output_shape = output_shape

        self.linear = nn.Linear(depth * 32, depth * 32)
        self.deconv_layers = nn.ModuleList()

        num_layers = len(kernels)
        for i, kernel in enumerate(kernels):
            if i != num_layers - 1:
                out_c = (2 ** (num_layers - i - 2)) * depth
                self.deconv_layers.append((kernel, out_c, True))
            else:
                self.deconv_layers.append((kernel, output_shape[-1], False))

    def forward(self, features):
        # features: [B,T,D]
        B, T, D = features.shape
        flat = features.view(B * T, D)

        x = self.linear(flat)  # [B*T, 32*depth]
        x = x.view(B * T, 32 * self.depth, 1, 1)

        for i, (kernel, out_c, intermediate) in enumerate(self.deconv_layers):
            deconv = nn.ConvTranspose2d(
                x.size(1), out_c, kernel_size=kernel, stride=2, padding=0
            ).to(x.device)
            x = deconv(x)
            if intermediate:
                x = F.relu(x)

        # x: [B*T,C,H,W]
        # reshape back to [B,T,H,W,C]
        C = self.output_shape[0]
        H, W = x.size(2), x.size(3)
        x = x.view(B, T, C, H, W)

        loc = x
        scale = torch.ones_like(x)
        dist = Independent(Normal(loc, scale), len(self.output_shape))
        return dist


class TransitionDecoder(nn.Module):
    def __init__(
        self,
        output_sizes,
        dist,
    ):
        super(TransitionDecoder, self).__init__()
        self.dist_type = dist
        layers = []
        input_dim = output_sizes[0]
        for size in output_sizes[1:]:
            layers.append(nn.Linear(input_dim, size))
            layers.append(nn.ELU())
            input_dim = size
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor):
        B, T, D = features.shape
        x = features.view(B * T, D)
        x = self.mlp(x)
        x = x.view(B, T, D)

        if self.dist_type == "normal":
            loc = x
            scale = torch.ones_like(x)
            base_dist = Normal(loc, scale)
            dist = Independent(base_dist, 0)
        elif self.dist_type == "bernoulli":
            dist = Bernoulli(logits=x)
            dist = Independent(dist, 0)
        else:
            raise ValueError("Unknown distribution type.")

        return dist


def init_hidden_state(batch_size, stochastic_size, deterministic_size, device):
    return (
        torch.zeros(batch_size, stochastic_size, device=device),
        torch.zeros(batch_size, deterministic_size, device=device),
    )


class Prior(nn.Module):
    """
    The prior predicts a distribution over the next stochastic latent state
    given the previous latent state and action, but WITHOUT looking at the new observation.
    This allows "imagining" future states before we actually see them.
    """

    def __init__(self, action_size, stochastic_size, deterministic_size, hidden_size):
        super(Prior, self).__init__()
        self.h1 = nn.Linear(stochastic_size + action_size, deterministic_size)
        self.gru = nn.GRUCell(deterministic_size, deterministic_size)
        self.h2 = nn.Linear(deterministic_size, hidden_size)
        self.h3 = nn.Linear(hidden_size, stochastic_size * 2)

    def forward(self, prev_state, prev_action):
        stoch, det = prev_state
        # Concatenate the previous action and the previous stochastic state.
        # This forms the input to the model for predicting the next latent state.
        cat = torch.cat([prev_action, stoch], dim=-1)
        x = F.elu(self.h1(cat))
        det = self.gru(x, det)  # Recurrently update the deterministic state.
        x = F.elu(self.h2(det))
        # Final linear layer outputs twice as many values as the stochastic size:
        # half for the mean of the next stochastic state, half for the stddev.
        # The output shape: [..., 2 * stochastic_size]
        x = self.h3(x)
        mean, stddev = torch.chunk(x, 2, dim=-1)
        # Use softplus to ensure stddev > 0, and add a small offset to improve stability.
        stddev = F.softplus(stddev) + 0.1
        dist = torch.distributions.Independent(
            torch.distributions.Normal(mean, stddev), 1
        )
        stoch = dist.rsample()
        return dist, (stoch, det)


class Posterior(nn.Module):
    """
    The posterior refines the prediction by incorporating the actual observation.
    Given the previous state and the new observation, it produces a distribution over
    the next stochastic state that should match reality better than the prior alone.
    """

    def __init__(self, stochastic_size, deterministic_size, hidden_dim, latent_dim):
        super(Posterior, self).__init__()
        self.h1 = nn.Linear(deterministic_size + latent_dim, hidden_dim)
        self.h2 = nn.Linear(hidden_dim, stochastic_size * 2)

    def forward(self, prev_state, observation):
        _, det = prev_state
        # Concatenate the deterministic state and the current observation.
        # This gives the model direct "evidence" from the new observation.
        cat = torch.cat([det, observation], dim=-1)
        x = F.elu(self.h1(cat))
        x = self.h2(x)
        mean, stddev = torch.chunk(x, 2, dim=-1)
        stddev = F.softplus(stddev) + 0.1
        # Construct a Normal distribution representing the posterior belief of the next state.
        dist = torch.distributions.Independent(
            torch.distributions.Normal(mean, stddev), 1
        )
        # Sample the refined stochastic state from the posterior.
        stoch = dist.rsample()
        return dist, (stoch, det)


class RSSM(nn.Module):
    def __init__(self, config):
        super(RSSM, self).__init__()
        self.config = config
        self.prior = Prior(config["rssm"])
        self.posterior = Posterior(config["rssm"])
        self.stoch_size = config["rssm"]["stochastic_size"]
        self.det_size = config["rssm"]["deterministic_size"]
        self.horizon = config["imag_horizon"]

    def forward(self, prev_state, prev_action, observation):
        prior, state = self.prior(prev_state, prev_action)
        posterior, state = self.posterior(state, observation)
        return (prior, posterior), state

    def imagine(self, init_obs, actor, actions=None):
        """
        Using the prior model, we can "imagine" future states without seeing observations.
        This is useful for planning or training policies purely in latent space.

        If actions are not given, we sample them from a provided policy (actor).
        If actions are given, we just roll forward using those actions.
        """
        horizon = self.horizon if actions is None else actions.size(1)
        seq_dim = self.stoch_size + self.det_size
        sequence = torch.zeros(
            init_obs.size(0), horizon, seq_dim, device=init_obs.device
        )

        stoch, det = torch.split(init_obs, [self.stoch_size, self.det_size], dim=-1)
        state = (stoch, det)

        # Loop through each step, predicting forward in time using the prior.
        for t in range(horizon):
            if actions is None:
                with torch.no_grad():
                    dist = actor(torch.cat(state, dim=-1))
                action = dist.rsample()
            else:
                action = actions[:, t]
            # Use the prior to predict the next state given current state & chosen action.
            _, state = self.prior(state, action)
            sequence[:, t] = torch.cat(state, dim=-1)

        return sequence

    def observe(self, observations, actions):
        """
        Here we process a sequence of observed steps (observations and corresponding actions),
        updating the latent state at each step.

        We collect the prior and posterior distributions at each step. This allows us to later
        compute losses like KL divergence between prior and posterior to guide learning.
        We also return the latent features (concatenation of stochastic and deterministic states)
        at each time step for further processing.
        """
        priors = []
        posteriors = []
        batch_size, time_size = observations.size(0), observations.size(1)
        feat_dim = self.stoch_size + self.det_size
        features = torch.zeros(
            batch_size, time_size, feat_dim, device=observations.device
        )

        state = init_hidden_state(
            batch_size, self.stoch_size, self.det_size, device=observations.device
        )

        for t in range(time_size):
            (prior, posterior), state = self.forward(
                state, actions[:, t], observations[:, t]
            )
            # Extract mean and std from prior and posterior
            # prior and posterior are Independent(Normal(...),1)
            # mean: [B,D], stddev: [B,D]
            prior_mean = prior.base_dist.loc
            prior_std = prior.base_dist.scale
            post_mean = posterior.base_dist.loc
            post_std = posterior.base_dist.scale

            priors.append((prior_mean, prior_std))
            posteriors.append((post_mean, post_std))

            features[:, t] = torch.cat(state, dim=-1)

        def joint_mvn(dists):
            # dists is a list of (mean, std) pairs, each [B,D]
            # Stack along time: results in [B,T,D]
            means = torch.stack([m for (m, _) in dists], dim=1)  # [B,T,D]
            stds = torch.stack([s for (_, s) in dists], dim=1)  # [B,T,D]

            # We have [B,T,D], treat T and D as event dimensions.
            # reinterpreted_batch_ndims=2:
            # - By default, Normal is over [B,T,D]. B is batch, T,D are also batch dims.
            # - We want B as batch, and (T,D) as event. So we must re-interpret two dims as event dims.
            # However, PyTorch distributions only allow event dimensions at the end.
            # Current shape: [B,T,D]. B is batch, T and D are "batch" as well.
            # We want a single joint distribution for each element in B over T,D.
            # That means the last two dims (T,D) are event dims.
            # reinterpreted_batch_ndims counts from the right. For shape [B,T,D], B=batch_dim0, T=batch_dim1, D=batch_dim2.
            # We want to interpret T,D as event => 2 event dims from the right.
            dist = torch.distributions.Independent(
                torch.distributions.Normal(means, stds), reinterpreted_batch_ndims=2
            )
            return dist

        priors = joint_mvn(priors)
        posteriors = joint_mvn(posteriors)

        return (priors, posteriors), features


class WorldModel(nn.Module):
    def __init__(
        self,
        rssm,
        observation_encoder,
        observation_decoder,
        reward_decoder,
        terminal_decoder,
    ):
        super(WorldModel, self).__init__()
        self.rssm = rssm
        self.encoder = observation_encoder
        self.decoder = observation_decoder
        self.reward = reward_decoder
        self.terminal = terminal_decoder

    def forward(self, prev_state, prev_action, observation):
        encoded_obs = self.encoder(observation)
        (prior, posterior), state = self.rssm(prev_state, prev_action, encoded_obs)
        return (prior, posterior), state

    def generate_sequence(self, initial_features, actor, actions=None):
        features = self.rssm.imagine(initial_features, actor, actions)
        reward_dist = self.reward(features)  # Normal distribution
        terminal_dist = self.terminal(features)  # Bernoulli distribution
        return features, reward_dist, terminal_dist

    def observe_sequence(self, observations, actions):
        encoded_obs = self.encoder(observations)
        (prior, posterior), features = self.rssm.observe(encoded_obs, actions)
        reward_dist = self.reward(features)  # Normal distribution
        terminal_dist = self.terminal(features)  # Bernoulli distribution
        decoded_obs = self.decoder(features)  # Normal distribution
        return (prior, posterior), features, decoded_obs, reward_dist, terminal_dist


class Actor(nn.Module):
    def __init__(self, input_dim, action_space, min_stddev=1e-4):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.action_space = action_space
        self.min_stddev = min_stddev

        # Determine if action space is continuous or discrete
        self.is_continuous = (
            self.action_space.dtype == float and len(self.action_space.shape) == 1
        )
        # If continuous, output mean and std for each action dimension
        # If discrete, output logits for each action dimension
        if self.is_continuous:
            self.action_dim = self.action_space.shape[0]
            # We will produce mean and stddev for continuous actions
            # So final layer dimension: action_dim * 2
            self.final_out = self.action_dim * 2
        else:
            # Discrete action space: final_out = number of discrete actions
            self.action_dim = self.action_space.n
            self.final_out = self.action_dim

        self.hidden_sizes = [400, 400, 400, 400]

        layers = []
        prev_dim = self.input_dim
        for size in self.hidden_sizes:
            layer = nn.Linear(prev_dim, size)
            layers.append(layer)
            layers.append(nn.ELU())
            prev_dim = size
        layers.append(nn.Linear(prev_dim, self.final_out))
        self.mlp = nn.Sequential(*layers)

        # Precompute init_std for continuous actions
        if self.is_continuous:
            self.init_std = float(np.log(np.exp(5.0) - 1.0))

    def forward(self, features: torch.Tensor):
        x = self.mlp(features)

        if self.is_continuous:
            half = x.size(-1) // 2
            mu = x[..., :half]
            stddev_param = x[..., half:]
            stddev = F.softplus(stddev_param + self.init_std) + self.min_stddev
            mean = 5.0 * torch.tanh(mu / 5.0)
            base_dist = Normal(mean, stddev)
            dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])
            dist = Independent(dist, 1)
        else:
            dist = Categorical(logits=x)

        return dist


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
