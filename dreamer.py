import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import utils
import networks
from memory import ReplayBuffer
from logger import TrainingLogger
import gym


class Dreamer:
    def __init__(self, obs_shape, act_space, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_shape = obs_shape
        self.stoch_size = 30
        self.det_size = 200
        self.kl_scale = config.kl_scale
        self.clip = config.actor.clip
        self.lambda_ = config.lambda_
        self.discount = config.discount
        self.imag_horizon = config.imag_horizon
        self.path = f"weights/{config.task}"
        self.update_interval = config.update_interval
        self.update_steps = config.update_steps
        self.action_repeat = config.action_repeat
        self.prefill = config.prefill
        self.free_nats = torch.tensor(config.free_nats).to(self.device)

        if isinstance(act_space, gym.spaces.Discrete):  # For discrete action spaces
            self.n_actions = act_space.n
        else:  # For continuous action spaces / if isinstance(act_space, gym.spaces.Box)
            self.n_actions = act_space.shape[0]

        # print("Number of Actions: ", self.n_actions, type(self.n_actions))

        self.act_low, self.act_high = act_space.low, act_space.high
        self.random_action = lambda: act_space.sample()

        self.world_model = networks.WorldModel(self.n_actions).to(self.device)
        self.actor = networks.Actor(act_space).to(self.device)
        self.critic = networks.Critic().to(self.device)

        self.memory = ReplayBuffer(config.replay, self.device)
        self.logger = TrainingLogger(config.log_dir)

        self.state = self.init_state()
        self.step = 0

    def __call__(self, obs, training=True):
        if training:
            if self.warmup_episodes:
                return self.random_action()

            if self.time_to_learn:
                self.learn()

        prev_state, prev_action = self.state
        state, action = self.act(prev_state, prev_action, obs, training)
        self.state = state, action

        return np.clip(action.squeeze().cpu().numpy(), self.act_low, self.act_high)

    def init_state(self):
        stoch = torch.zeros(1, self.stoch_size).to(self.device)
        det = torch.zeros(1, self.det_size).to(self.device)
        action = torch.zeros((1, self.n_actions)).to(self.device)
        return (stoch, det), action

    def act(self, prev_state, prev_action, obs, training=True):
        # Add batch and time dimensions to single observation
        obs = obs.unsqueeze(0).unsqueeze(0).to(self.device)
        # print("Dreamer Prev State: ", prev_state[0].shape, prev_state[1].shape)
        # print("Dreamer Prev Action: ", prev_action.shape)
        # print("Dreamer Obs Shape: ", obs.shape, obs.dtype)

        # Process observation to get latent state
        _, current_state = self.world_model(prev_state, prev_action, obs)

        # Combine stochastic and deterministic parts into a single features vector.
        features = torch.cat(current_state, dim=-1)

        # Compute action distribution
        policy = self.actor(features)

        # Sample action during training for exploration; use mode for evaluation.
        action = policy.sample() if training else policy.mode()
        # print("Dreamer Action:", action.shape)
        return current_state, action.unsqueeze(0)

    def learn(self):
        report = defaultdict(float)
        steps = self.update_steps
        batches = self.memory.sample(steps)

        for batch in tqdm(batches, total=steps):
            outputs = self.update(batch)

            for k, v in outputs.items():
                report[k] += float(v) / steps

        self.logger.log_metrics(report, self.step)

    def update(self, batch):
        world_metrics, features = self.update_world_model(batch)
        actor_metrics, imag_features, lambda_values = self.update_actor(
            features.detach()
        )
        critic_metrics = self.update_critic(
            imag_features.detach(), lambda_values.detach()
        )

        metrics = {**world_metrics, **actor_metrics, **critic_metrics}
        return metrics

    def update_world_model(self, batch):
        """
        Update the world model parameters:
        Minimizes KL divergence between prior and posterior,
        Maximizes likelihood of reconstructing observations, rewards, and terminal signals.
        """
        obs = batch["observation"]
        act = batch["action"]
        rew = batch["reward"]
        done = batch["done"]

        (prior, posterior), features, decoded, reward, terminal = (
            self.world_model.observe(obs, act)
        )

        # Compute KL divergence between prior and posterior
        kl_div = torch.maximum(
            torch.distributions.kl.kl_divergence(posterior, prior).mean(),
            self.free_nats,
        )

        # Compute reconstruction loss
        obs_loss = decoded.log_prob(obs).mean()

        # Compute reward prediction loss
        # print("Reward:", rew.shape)
        rew_loss = reward.log_prob(rew.unsqueeze(-1)).mean()

        # Compute terminal prediction loss
        term_loss = terminal.log_prob(done.unsqueeze(-1)).mean()

        # Overall world model loss: encourages good reconstruction and small KL.
        loss = self.kl_scale * kl_div - obs_loss - rew_loss - term_loss

        self.world_model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.clip)
        grad_norm = utils.global_norm([p.grad for p in self.world_model.parameters()])
        self.world_model.optimizer.step()

        metrics = {
            "agent/world/kl": kl_div,
            "agent/world/post_entropy": posterior.entropy().mean(),
            "agent/world/prior_entropy": prior.entropy().mean(),
            "agent/world/log_p_observation": -obs_loss,
            "agent/world/log_p_reward": -rew_loss,
            "agent/world/log_p_terminal": -term_loss,
            "agent/world/grad_norm": grad_norm,
        }

        return metrics, features

    def update_actor(self, features):
        """
        Update the actor parameters using imagined rollouts from the model:
        1. Generate imaginary futures from current features using the policy.
        2. Compute lambda-returns from rewards and value predictions (critic).
        3. Maximize these returns by adjusting the actor to produce better actions.
        """
        flat_feats = features.view(-1, features.shape[-1])

        # Generate imaginary futures using the actor
        imag_feats, rewards_dist, terminals_dist = self.world_model.imagine(
            flat_feats, self.actor
        )

        rewards = rewards_dist.rsample()  # Use .sample() for stochastic updates
        terminals = terminals_dist.sample()
        next_values = self.critic(
            imag_feats[:, 1:]
        ).rsample()  # Extract value estimates

        # Compute lambda-returns
        lambda_values = utils.compute_lambda_values(
            next_values, rewards, terminals, self.discount, self.lambda_
        )

        discount = utils.discount(self.discount, self.imag_horizon - 1)
        loss = (-lambda_values * discount.to(self.device)).mean()

        self.actor.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip)
        grad_norm = utils.global_norm([p.grad for p in self.actor.parameters()])
        self.actor.optimizer.step()

        entropy = self.actor(features[:, 0]).entropy().mean()

        metrics = {
            "agent/actor/loss": loss,
            "agent/actor/grad_norm": grad_norm,
            "agent/actor/entropy": entropy,
        }

        return metrics, imag_feats, lambda_values

    def update_critic(self, features, lambda_values):
        """
        Update the critic parameters:
        Critic predicts value distribution. We train it with the lambda-returns computed above.
        By maximizing log probability of lambda_values, the critic learns a good value function.
        """
        values = self.critic(features[:, :-1])
        discount = utils.discount(self.discount, self.imag_horizon - 1)

        # print("Values:",values.shape)
        # print("Targets:", targets.shape)
        # print("Discount:", discount.shape)
        loss = -(values.log_prob(lambda_values) * discount.to(self.device)).mean()

        self.critic.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip)
        grad_norm = utils.global_norm([p.grad for p in self.critic.parameters()])
        self.critic.optimizer.step()

        metrics = {
            "agent/critic/loss": loss,
            "agent/critic/grad_norm": grad_norm,
        }

        return metrics

    def remember(self, transition):
        # print("Observing")
        obs = transition["observation"]
        act = transition["action"]
        rew = transition["reward"]
        done = transition["terminal"]

        self.memory.store(obs, act, rew, done)
        if done:
            self.state = self.init_state()

        self.step += self.action_repeat

    @property
    def warmup_episodes(self):
        return self.step <= self.prefill

    @property
    def time_to_learn(self):
        return self.step % self.update_interval == 0

    def save_checkpoint(self):
        torch.save(self.world_model.state_dict(), f"{self.path}_world_model.pt")
        torch.save(self.actor.state_dict(), f"{self.path}_actor.pt")
        torch.save(self.critic.state_dict(), f"{self.path}_critic.pt")

    def load_checkpoint(self):
        self.world_model.load_state_dict(
            torch.load(f"{self.path}_world_model.pt", weights_only=True)
        )
        self.actor.load_state_dict(
            torch.load(f"{self.path}_actor.pt", weights_only=True)
        )
        self.critic.load_state_dict(
            torch.load(f"{self.path}_critic.pt", weights_only=True)
        )
