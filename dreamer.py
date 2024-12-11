import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

import utils
import networks
from memory import ReplayBuffer
from logger import TrainingLogger
import gym


class Dreamer:
    def __init__(self, obs_shape, act_space, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.path = f"weights/{config.task}"

        if isinstance(act_space, gym.spaces.Box):  # For continuous action spaces
            self.n_actions = act_space.shape[0]
        elif isinstance(act_space, gym.spaces.Discrete):  # For discrete action spaces
            self.n_actions = act_space.n

        self.act_low, self.act_high = act_space.low, act_space.high
        self.random_action = lambda: act_space.sample()

        self.world_model = networks.WorldModel(obs_shape, self.n_actions, config).to(
            self.device
        )
        self.actor = networks.Actor(act_space, config).to(self.device)
        self.critic = networks.Critic(config).to(self.device)

        self.memory = ReplayBuffer(config.replay, self.device)
        self.logger = TrainingLogger(config)

        self.state = self.init_state()
        self.step = 0

    def __call__(self, obs, training=True):
        if training:
            if self.warmup_episodes:
                return self.random_action()

            elif self.update_interval:
                self.learn()

        prev_state, prev_action = self.state
        state, action = self.act(prev_state, prev_action, obs, training)
        self.state = state, action

        return torch.clip(action, self.act_low, self.act_high).cpu().numpy()

    def init_weights(self):
        pass

    def init_state(self):
        stoch = torch.zeros(1, self.config.rssm.stochastic_size)
        det = torch.zeros(1, self.config.rssm.deterministic_size)
        action = torch.zeros((1, self.n_actions))
        return (stoch, det), action

    def act(self, prev_state, prev_action, obs, training=True):
        # Add batch and time dimensions to single observation
        obs = obs.unsqueeze(0).unsqueeze(0)
        # print("Dreamer Prev State: ", len(prev_state))
        # print("Dreamer Prev State: ", prev_state[0].shape, prev_state[1].shape)
        # print("Dreamer Prev Action: ", prev_action)
        # print("Dreamer Obs Shape: ", obs.shape, obs.dtype)

        # Process observation to get latent state
        _, current_state = self.world_model(prev_state, prev_action, obs)

        # Combine stochastic and deterministic parts into a single features vector.
        features = torch.cat(current_state, dim=-1)

        # Compute action distribution
        policy = self.actor(features)

        # Sample action during training for exploration; use mode for evaluation.
        action = policy.sample() if training else policy.mode()
        return current_state, action.squeeze(0)

    def learn(self):
        report = defaultdict(float)
        steps = self.config.update_steps
        batches = self.memory.sample(steps)

        for batch in tqdm(batches, total=steps):
            outputs = self.update(batch)

            for k, v in outputs.items():
                report[k] += float(v.item()) / steps

        self.logger.log_metrics(report, self.step)

    def update(self, batch):
        world_metrics, features = self.update_world_model(batch)
        actor_metrics, imag_features, lambda_values = self.update_actor(features)
        critic_metrics = self.update_critic(imag_features, lambda_values)

        metrics = {**world_metrics, **actor_metrics, **critic_metrics}
        return metrics

    def update_world_model(self, batch):
        """
        Update the world model parameters:
        Minimizes KL divergence between prior and posterior,
        Maximizes likelihood of reconstructing observations, rewards, and terminal signals.
        """
        (prior, posterior), features, decoded, reward, terminal = (
            self.world_model.observe(batch["observation"], batch["action"])
        )

        # Compute KL divergence between prior and posterior
        kl_div = torch.maximum(F.kl_div(posterior, prior).mean(), self.config.free_nats)

        # Compute reconstruction loss
        obs_loss = decoded.log_prob(batch["observation"]).mean()

        # Compute reward prediction loss
        rew_loss = reward.log_prob(batch["reward"]).mean()

        # Compute terminal prediction loss
        term_loss = terminal.log_prob(batch["done"]).mean()

        # Overall world model loss: encourages good reconstruction and small KL.
        loss = self.config.kl_scale * kl_div - obs_loss - rew_loss - term_loss

        self.world_model.optimizer.zero_grad()
        loss.backward()
        torch.gradient.clip_grad_norm_(
            self.world_model.parameters(), self.config.model_opt.clip
        )
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
        imag_feats, rewards, dones = self.world_model.imagine(flat_feats, self.actor)

        # Predict values for imagined futures
        next_values = self.critic(imag_feats[:, 1:]).mean()

        # Compute lambda-returns
        lambda_values = utils.compute_lambda_values(
            next_values, rewards, dones, self.config.discount, self.config.lambda_
        )

        discount = utils.discount(self.c.discount, self.c.imag_horizon - 1)
        loss = (-lambda_values * discount).mean()

        self.actor.optimizer.zero_grad()
        loss.backward()
        torch.gradient.clip_grad_norm_(self.actor.parameters(), self.config.actor.clip)
        grad_norm = utils.global_norm([p.grad for p in self.actor.parameters()])
        self.actor.optimizer.step()

        entropy = self.actor(features[:, 0]).entropy().mean()

        metrics = {
            "agent/actor/loss": loss,
            "agent/actor/grad_norm": grad_norm,
            "agent/actor/entropy": entropy,
        }

        return metrics, features, lambda_values

    def update_critic(self, features, lambda_values):
        """
        Update the critic parameters:
        Critic predicts value distribution. We train it with the lambda-returns computed above.
        By maximizing log probability of lambda_values, the critic learns a good value function.
        """
        values = self.critic(features[:, :-1])
        targets = lambda_values.detach()
        discount = utils.discount(self.config.discount, self.config.imag_horizon - 1)
        loss = -(values.log_prob(targets) * discount).mean()

        self.critic.optimizer.zero_grad()
        loss.backward()
        torch.gradient.clip_grad_norm_(
            self.critic.parameters(), self.config.critic.clip
        )
        grad_norm = utils.global_norm([p.grad for p in self.critic.parameters()])
        self.critic.optimizer.step()

        metrics = {
            "agent/critic/loss": loss,
            "agent/critic/grad_norm": grad_norm,
        }

        return metrics

    def observe(self, transition):
        obs = transition["observation"]
        act = transition["action"]
        rew = transition["reward"]
        done = transition["terminal"]

        self.memory.store(obs, act, rew, done)
        if done:
            self.state = self.init_state()

        self.step += self.config.action_repeat

    @property
    def warmup_episodes(self):
        return self.step <= self.config.prefill

    @property
    def update_interval(self):
        return self.step % self.config.update_interval == 0

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
