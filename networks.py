import numpy as np
import gym
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

from utils import fan_initializer


class ObservationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ELU(),
        )

        for layer in self.convs:
            if isinstance(layer, nn.Conv2d):
                fan_initializer(1.0, "fan_avg", "uniform")(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.convs(x)
        x = x.view(B, T, -1)
        return x


class ObservationDecoder(nn.Module):
    def __init__(self):
        """
        Decoder to reconstruct observations from the 230-dimensional RSSM latent state,
        predicting only the mean of a normal distribution with fixed variance 1.0.
        """
        super().__init__()
        self.fc = nn.Linear(
            230, 256 * 4 * 4
        )  # Project to 256 channels with 1x1 spatial size
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.fc.bias)

        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.ELU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.ELU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=6, stride=2, padding=2, output_padding=0
            ),
            nn.ELU(),
            nn.ConvTranspose2d(
                32, 3, kernel_size=6, stride=2, padding=2, output_padding=0
            ),
        )

        for layer in self.deconvs:
            if isinstance(layer, nn.ConvTranspose2d):
                fan_initializer(1.0, "fan_avg", "uniform")(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        B, T, D = x.shape
        # Project latent state to match ConvTranspose2d input shape
        # print("Decoder input:", x.shape)
        x = x.view(B * T, D)
        x = self.fc(x)  # [B x T, 230] -> [B x T, 256 * 4 * 4]
        # print("Decoder upscale:", x.shape)
        x = x.view(B * T, 256, 4, 4)  # [B x T, 256, 4, 4]
        x = self.deconvs(x)  # [B x T, 3, 64, 64]
        mean = x.view(B, T, 3, 64, 64)  # [B, T, 3, 64, 64]
        # Return a Normal distribution with fixed variance
        # reinterpreted_batch_ndims=3 treats the last three dimensions (channels, height, width)
        # as part of each independent event in the distribution.
        # This means the distribution models independent pixel values, instead of treating
        # each pixel as a separate batch.
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, 1.0), reinterpreted_batch_ndims=3
        )


class TransitionDecoder(nn.Module):
    def __init__(
        self,
        dist,
    ):
        super(TransitionDecoder, self).__init__()
        self.dist_type = dist
        self.input_dim = 230
        self.hidden_layers = [300, 300, 300]
        layers = []
        prev_dim = self.input_dim
        for size in self.hidden_layers:
            layer = nn.Linear(prev_dim, size)
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            layers.append(nn.ELU())
            prev_dim = size
        layers.append(nn.Linear(size, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features):
        B, T, D = features.shape
        x = features.view(B * T, D)
        x = self.mlp(x)
        x = x.view(B, T, 1)

        if self.dist_type == "normal":
            mu = x
            std = torch.ones_like(x)
            dist = Normal(mu, std)
            dist = Independent(dist, 0)
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

    def __init__(self, n_actions):
        super(Prior, self).__init__()
        self.stoch_size = 30
        self.det_size = 200
        self.hidden_size = 200
        self.input_shape = self.stoch_size + int(n_actions)
        self.h1 = nn.Linear(self.input_shape, self.det_size)
        self.gru = nn.GRUCell(self.det_size, self.det_size)
        self.h2 = nn.Linear(self.det_size, self.hidden_size)
        self.h3 = nn.Linear(self.hidden_size, self.stoch_size * 2)

    def forward(self, prev_state, prev_action):
        stoch, det = prev_state
        # Concatenate the previous action and the previous stochastic state.
        # This forms the input to the model for predicting the next latent state.
        # print("Prior Prev Action:", prev_action.shape)
        # print("Prior Stoch:", stoch.shape)
        # print("Stoch:", stoch)
        cat = torch.cat([prev_action, stoch], dim=-1)
        # print("Prior Cat:", cat.shape)
        # print("Prior Input shape:", self.input_shape)
        x = F.elu(self.h1(cat))
        det = self.gru(x, det)  # Recurrently update the deterministic state.
        # print("Prior Det:", det.shape)
        x = F.elu(self.h2(det))
        # Final linear layer outputs twice as many values as the stochastic size:
        # half for the mean of the next stochastic state, half for the stddev.
        # The output shape: [..., 2 * stochastic_size]
        x = self.h3(x)
        # print("Prior x:", x.shape)
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

    def __init__(self):
        super(Posterior, self).__init__()
        self.stoch_size = 30
        self.det_size = 200
        self.hidden_size = 200
        self.obs_size = 1024
        self.h1 = nn.Linear(self.obs_size + self.det_size, self.hidden_size)
        self.h2 = nn.Linear(self.hidden_size, self.stoch_size * 2)

    def forward(self, prev_state, observation):
        _, det = prev_state
        observation = observation.squeeze(1)
        # Concatenate the deterministic state and the current observation.
        # This gives the model direct "evidence" from the new observation.
        # print("\nPosterior Det:", det.shape)
        # print("Posterior Embed:", observation.shape)
        cat = torch.cat([det, observation], dim=-1)
        # print("Cat:", cat.shape)
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
    def __init__(self, n_actions):
        super(RSSM, self).__init__()
        self.prior = Prior(n_actions)
        self.posterior = Posterior()
        self.stoch_size = 30
        self.det_size = 200
        self.horizon = 15

    def forward(self, prev_state, prev_action, observation):
        # print("\nRSSM Embed shape: ", observation.shape, observation.dtype)
        # print("RSSM Prev state:", prev_state[0].shape, prev_state[1].shape)
        # print("RSSM Prev action:", prev_action.shape)
        prior, state = self.prior(prev_state, prev_action)
        posterior, state = self.posterior(state, observation)
        return (prior, posterior), state

    def imagine(self, init_obs, actor=None, actions=None):
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
        # print("\nImagine Stoch shape:", stoch.shape)

        # Loop through each step, predicting forward in time using the prior.
        for t in range(horizon):
            if actions is None:
                with torch.no_grad():
                    # print(type(actor))
                    dist = actor(torch.cat(state, dim=-1))
                # Not handling this correctly, should be:
                # action = dist.rsample()
                # for continuous spaces
                action = dist.rsample()
            else:
                action = actions[:, t]
            # action = action.unsqueeze(-1)
            # print("Imagine Action shape:", action.shape)
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
    def __init__(self, n_actions, lr=6e-4, eps=1e-7):
        super(WorldModel, self).__init__()
        self.encoder = ObservationEncoder()
        self.decoder = ObservationDecoder()
        self.reward = TransitionDecoder("normal")
        self.terminal = TransitionDecoder("bernoulli")
        self.rssm = RSSM(n_actions)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(lr),
            eps=float(eps),
        )

    def forward(self, prev_state, prev_action, observation):
        encoded_obs = self.encoder(observation)
        # print("\nWorld Obs shape: ", observation.shape, observation.dtype)
        # print("World Embed shape: ", encoded_obs.shape, encoded_obs.dtype)
        # print("World Prev state:", prev_state[0].shape, prev_state[1].shape)
        # print("World Prev action:", prev_action.shape)
        (prior, posterior), state = self.rssm(prev_state, prev_action, encoded_obs)
        return (prior, posterior), state

    def imagine(self, initial_features, actor=None, actions=None):
        # print(type(actor))
        features = self.rssm.imagine(initial_features, actor, actions)
        reward_dist = self.reward(features)  # Normal distribution
        terminal_dist = self.terminal(features)  # Bernoulli distribution
        return features, reward_dist, terminal_dist

    def observe(self, observations, actions):
        # print("Obs shape:", observations.shape)
        encoded_obs = self.encoder(observations)
        # print("Encoded obs shape:", encoded_obs.shape)
        (prior, posterior), features = self.rssm.observe(encoded_obs, actions)
        # print("Latent shape:", features.shape)
        reward_dist = self.reward(features)  # Normal distribution
        terminal_dist = self.terminal(features)  # Bernoulli distribution
        decoded_obs = self.decoder(features)  # Normal distribution
        return (prior, posterior), features, decoded_obs, reward_dist, terminal_dist

    def decode(self, features):
        return self.decoder(features)


class Actor(nn.Module):
    def __init__(self, n_actions, is_continuous, lr=8e-5, eps=1e-7):
        super(Actor, self).__init__()
        self.input_dim = 230
        self.hidden_layers = [300, 300, 300]
        self.is_continuous = is_continuous

        # If continuous, output mean and std for each action dimension
        # If discrete, output logits for each action dimension
        if self.is_continuous:
            # print("Continuous action space")
            self.init_std = float(np.log(np.exp(5.0) - 1.0))
            self.final_out = n_actions * 2
        else: 
            # isinstance(action_space, gym.spaces.Discrete):
            # print("Discrete action space")
            self.final_out = n_actions

        layers = []
        prev_dim = self.input_dim
        for size in self.hidden_layers:
            layer = nn.Linear(prev_dim, size)
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            layers.append(nn.ELU())
            prev_dim = size
        layers.append(nn.Linear(prev_dim, self.final_out))
        self.mlp = nn.Sequential(*layers)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=float(lr), eps=float(eps)
        )

    def forward(self, x):
        x = self.mlp(x)

        if self.is_continuous:
            half = x.size(-1) // 2
            mu = x[..., :half]
            stddev_param = x[..., half:]
            stddev = F.softplus(stddev_param + self.init_std) + 1e-6
            mean = 5.0 * torch.tanh(mu / 5.0)
            dist = Normal(mean, stddev)
            # dist = TransformedDistribution(dist, [TanhTransform(cache_size=1)])
            dist = Independent(dist, 1)
        else:
            dist = Categorical(logits=x)

        return dist


class Critic(nn.Module):
    def __init__(self, lr=8e-5, eps=1e-7):
        super(Critic, self).__init__()
        self.input_dim = 230
        self.hidden_layers = [300, 300, 300]

        layers = []
        prev_dim = self.input_dim
        for size in self.hidden_layers:
            layer = nn.Linear(prev_dim, size)
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            layers.append(nn.ELU())
            prev_dim = size

        # Predict mean and stddev for a Normal distribution
        self.mean_layer = nn.Linear(prev_dim, 1)
        self.std_layer = nn.Linear(prev_dim, 1)
        self.mlp = nn.Sequential(*layers)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=float(lr), eps=float(eps)
        )

    def forward(self, x):
        x = self.mlp(x)
        mean = self.mean_layer(x)
        std = F.softplus(self.std_layer(x)) + 1e-6  # Ensure stddev is positive
        return Independent(Normal(mean, std), 1)
