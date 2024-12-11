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
    def __init__(self, input_shape, config):
        super(ObservationEncoder, self).__init__()
        self.depth = config.depth
        self.kernels = config.kernels
        self.input_shape = input_shape

        conv_layers = []
        c_in = self.input_shape[0]
        for i, kernel in enumerate(self.kernels):
            c_out = (2**i) * self.depth
            conv = nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=2, padding=0)
            conv_layers.append(conv)
            c_in = c_out
        self.convs = nn.ModuleList(conv_layers)

    def forward(self, x):
        print(x.shape)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        for conv in self.convs:
            x = F.relu(conv(x))

        x = x.view(B, T, -1)
        return x


class ObservationDecoder(nn.Module):
    def __init__(self, output_shape, config):
        super(ObservationDecoder, self).__init__()
        self.depth = config.depth
        self.kernels = [4, 4, 4, 4]  # All layers use these parameters
        self.output_shape = output_shape

        # Input is 230-d RSSM state
        # Map from 230 -> 4096 (256*4*4)
        self.linear = nn.Linear(230, 256 * 4 * 4)
        self.deconv_layers = nn.ModuleList()

        in_channels = 256
        # Each layer: kernel=4, stride=2, padding=1 doubles spatial dims
        for i in range(len(self.kernels)):
            if i < len(self.kernels) - 1:
                out_c = in_channels // 2
                layer = nn.ConvTranspose2d(
                    in_channels, out_c, kernel_size=4, stride=2, padding=1
                )
                in_channels = out_c
            else:
                out_c = self.output_shape[0]
                layer = nn.ConvTranspose2d(
                    in_channels, out_c, kernel_size=4, stride=2, padding=1
                )
            self.deconv_layers.append(layer)

    def forward(self, features):
        B, T, D = features.shape
        # features should be [B,T,230]
        x = self.linear(features.view(B * T, D))
        x = x.view(B * T, 256, 4, 4)
        for i, layer in enumerate(self.deconv_layers):
            x = layer(x)
            # Apply nonlinearity to all but the last layer
            if i < len(self.deconv_layers) - 1:
                x = F.elu(x)

        # Final shape: (B,T,C,H,W)
        C, H, W = x.size(1), x.size(2), x.size(3)
        x = x.view(B, T, C, H, W)
        loc = x
        scale = torch.ones_like(x)
        dist = Independent(Normal(loc, scale), len(self.output_shape))
        return dist


class TransitionDecoder(nn.Module):
    def __init__(
        self,
        config,
        dist,
    ):
        super(TransitionDecoder, self).__init__()
        self.dist_type = dist
        self.output_sizes = config.output_sizes
        # print("Layer sizes:", self.output_sizes)
        layers = []
        input_dim = self.output_sizes[0]
        for size in self.output_sizes[1:]:
            layers.append(nn.Linear(input_dim, size))
            layers.append(nn.ELU())
            input_dim = size
        self.mlp = nn.Sequential(*layers)

    def forward(self, features):
        # print(features.shape)
        B, T, D = features.shape
        x = features.view(B * T, D)
        # print(x.shape)
        x = self.mlp(x)
        x = x.view(B, T, self.output_sizes[-1])

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

    def __init__(self, n_actions, config):
        super(Prior, self).__init__()
        self.stoch_size = config.rssm.stochastic_size
        self.det_size = config.rssm.deterministic_size
        self.hidden_size = config.rssm.hidden_size
        self.h1 = nn.Linear(self.stoch_size + n_actions, self.det_size)
        self.gru = nn.GRUCell(self.det_size, self.det_size)
        self.h2 = nn.Linear(self.det_size, self.hidden_size)
        self.h3 = nn.Linear(self.hidden_size, self.stoch_size * 2)

    def forward(self, prev_state, prev_action):
        stoch, det = prev_state
        # Concatenate the previous action and the previous stochastic state.
        # This forms the input to the model for predicting the next latent state.
        print("Prev action:", prev_action.shape)
        print("Stoch:", stoch)
        print("Stoch:", stoch.shape)
        cat = torch.cat([prev_action, stoch], dim=-1)
        print("Cat:", cat.shape)
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

    def __init__(self, config):
        super(Posterior, self).__init__()
        self.stoch_size = config.rssm.stochastic_size
        self.det_size = config.rssm.deterministic_size
        self.hidden_size = config.rssm.hidden_size
        self.h1 = nn.Linear(1224, self.det_size + self.stoch_size)
        self.h2 = nn.Linear(self.det_size + self.stoch_size, self.hidden_size)
        self.h3 = nn.Linear(self.hidden_size, self.stoch_size * 2)

    def forward(self, prev_state, observation):
        _, det = prev_state
        # Concatenate the deterministic state and the current observation.
        # This gives the model direct "evidence" from the new observation.
        # print("Det:", det.shape)
        # print("Obs:", observation.shape)
        cat = torch.cat([det, observation], dim=-1)
        # print("Cat:", cat.shape)
        x = F.elu(self.h1(cat))
        x = F.elu(self.h2(x))
        x = self.h3(x)
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
    def __init__(self, n_actions, config):
        super(RSSM, self).__init__()
        self.prior = Prior(n_actions, config)
        self.posterior = Posterior(config)
        self.stoch_size = config.rssm.stochastic_size
        self.det_size = config.rssm.deterministic_size
        self.horizon = config.imag_horizon

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
        # print("Batch size:", batch_size)
        # print("Time size:", time_size)
        # print("Observations shape:", observations.shape)
        feat_dim = self.stoch_size + self.det_size
        features = torch.zeros(
            batch_size, time_size, feat_dim, device=observations.device
        )

        state = init_hidden_state(
            batch_size, self.stoch_size, self.det_size, device=observations.device
        )

        for t in range(time_size):
            # Compute (prior, posterior) and update the state using the current observation & action
            (prior, posterior), state = self.forward(
                state, actions[:, t], observations[:, t]
            )

            # Store the mean/std of the prior and posterior distributions.
            # This will be used to construct a joint distribution over time.
            prior_mean = prior.base_dist.loc
            prior_std = prior.base_dist.scale
            post_mean = posterior.base_dist.loc
            post_std = posterior.base_dist.scale

            priors.append((prior_mean, prior_std))
            posteriors.append((post_mean, post_std))

            # Store the latent features at this time step
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
        obs_shape,
        act_space,
        config,
    ):
        super(WorldModel, self).__init__()
        self.rssm = RSSM(act_space, config)
        self.encoder = ObservationEncoder(obs_shape, config.encoder)
        self.decoder = ObservationDecoder(obs_shape, config.decoder)
        self.reward = TransitionDecoder(config.reward, "normal")
        self.terminal = TransitionDecoder(config.terminal, "bernoulli")

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(config.model_opt.lr),
            eps=float(config.model_opt.eps),
        )

    def forward(self, prev_state, prev_action, observation):
        encoded_obs = self.encoder(observation)
        (prior, posterior), state = self.rssm(prev_state, prev_action, encoded_obs)
        return (prior, posterior), state

    def imagine(self, initial_features, actor, actions=None):
        features = self.rssm.imagine(initial_features, actor, actions)
        reward_dist = self.reward(features)  # Normal distribution
        terminal_dist = self.terminal(features)  # Bernoulli distribution
        return features, reward_dist, terminal_dist

    def observe(self, observations, actions):
        encoded_obs = self.encoder(observations)
        print("Encoded obs shape:", encoded_obs.shape)
        (prior, posterior), features = self.rssm.observe(encoded_obs, actions)
        print("State shape:", features.shape)
        reward_dist = self.reward(features)  # Normal distribution
        terminal_dist = self.terminal(features)  # Bernoulli distribution
        decoded_obs = self.decoder(features)  # Normal distribution
        return (prior, posterior), features, decoded_obs, reward_dist, terminal_dist


class Actor(nn.Module):
    def __init__(self, action_space, config):
        super(Actor, self).__init__()
        self.action_space = action_space
        self.input_dim = config.rssm.stochastic_size + config.rssm.deterministic_size
        self.min_stddev = config.actor.min_stddev
        self.hidden_sizes = config.actor.output_sizes

        self.init_layers()

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=float(config.actor.lr), eps=float(config.actor.eps)
        )

    def init_layers(self):
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

    def forward(self, x):
        x = self.mlp(x)

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
    def __init__(self, config):
        super(Critic, self).__init__()
        self.hidden_sizes = config.critic.output_sizes
        self.input_dim = config.rssm.stochastic_size + config.rssm.deterministic_size

        self.init_layers()

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=float(config.critic.lr), eps=float(config.critic.eps)
        )

    def init_layers(self):
        layers = []
        prev_dim = self.input_dim
        for size in self.hidden_sizes:
            layer = nn.Linear(prev_dim, size)
            layers.append(layer)
            layers.append(nn.ELU())
            prev_dim = size
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
