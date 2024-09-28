import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from utils import gumbel_softmax, symlog, symexp, ObsNormalizer, ReplayBuffer
from networks import WorldModel, Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DreamerV3:
    def __init__(self, obs_shape, act_dim, is_image, config):
        self.config = config
        self.obs_shape = obs_shape
        self.act_dim = act_dim
        self.is_image = is_image

        self.world_model = WorldModel(obs_shape, act_dim, is_image, config).to(device)
        self.actor = Actor(config.hidden_dim, act_dim, config).to(device)
        self.critic = Critic(config.hidden_dim, config).to(device)
        self.target_critic = Critic(config.hidden_dim, config).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.obs_normalizer = ObsNormalizer(obs_shape)

        self.world_optimizer = optim.AdamW(
            self.world_model.parameters(), lr=config.world_lr, weight_decay=config.weight_decay
        )
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(), lr=config.actor_lr, weight_decay=config.weight_decay
        )
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(), lr=config.critic_lr, weight_decay=config.weight_decay
        )

        self.replay_buffer = ReplayBuffer(config.replay_buffer_capacity, obs_shape, act_dim)

        # Hidden states
        self.h = None
        self.reset_hidden_states()

        # Temperature for Gumbel-Softmax
        self.temperature = config.init_temperature

        # Initialize networks
        self.world_model.apply(self.init_weights)
        self.actor.apply(self.init_weights)
        self.critic.apply(self.init_weights)
        self.target_critic.apply(self.init_weights)

        # Add observation normalizer if not using image observations
        if not is_image:
            self.obs_normalizer = ObsNormalizer(obs_shape)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            fan_in = m.weight.shape[1]
            fan_out = m.weight.shape[0]
            limit = np.sqrt(6 / (fan_in + fan_out))
            nn.init.uniform_(m.weight, -limit, limit)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRUCell):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def reset_hidden_states(self):
        self.h = torch.zeros(1, 1, self.config.hidden_dim, device=device)

    def update_world_model(self):
        try:
            batch = self.replay_buffer.sample_batch(
                self.config.batch_size, self.config.seq_len
            )
        except ValueError:
            return  # Not enough data to train

        obs_seq, act_seq, rew_seq, _, _ = batch

        outputs = self.world_model(obs_seq, act_seq, tau=self.temperature)
        recon_obs = outputs["recon_obs"]
        pred_reward = outputs["pred_reward"]
        kl_loss = outputs["kl_loss"]

        # Reconstruction loss
        if self.is_image:
            recon_loss = F.mse_loss(recon_obs, obs_seq, reduction='none')
            recon_loss = recon_loss.mean(dim=[2, 3, 4])  # Mean over image dimensions
        else:
            recon_loss = F.mse_loss(recon_obs, obs_seq, reduction='none')
            recon_loss = recon_loss.mean(dim=2)  # Mean over observation dimensions

        recon_loss = recon_loss.mean()  # Mean over batch and sequence length

        # Reward prediction loss
        reward_loss = F.mse_loss(pred_reward, symlog(rew_seq), reduction='none')
        reward_loss = reward_loss.mean()

        # Total loss
        loss_world = recon_loss + reward_loss + self.config.kl_scale * kl_loss

        self.world_optimizer.zero_grad()
        loss_world.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.max_grad_norm)
        self.world_optimizer.step()

        return loss_world.item()

    def update_actor_and_critic(self):
        # Imagined rollouts
        try:
            batch = self.replay_buffer.sample_batch(self.config.batch_size, 1)
        except ValueError:
            return  # Not enough data to train

        obs_seq, act_seq, _, _, _ = batch
        obs = obs_seq[:, 0]

        # Initialize hidden state with zeros
        imag_h = torch.zeros(1, self.config.batch_size, self.config.hidden_dim, device=device)

        # Encode observation
        obs_encoded = self.world_model.obs_encoder(obs)

        # Initialize imag_s with posterior sample from initial obs
        posterior_input = torch.cat([imag_h.squeeze(0), obs_encoded], dim=-1)
        posterior_logits = self.world_model.posterior_net(posterior_input)
        posterior_logits = posterior_logits.view(
            self.config.batch_size, self.config.latent_dim, self.config.latent_categories
        )
        imag_s = gumbel_softmax(posterior_logits, tau=self.temperature, hard=False).view(self.config.batch_size, -1)

        imag_states = []
        imag_rewards = []
        imag_action_log_probs = []

        for _ in range(self.config.imagination_horizon):
            imag_states.append(imag_h.squeeze(0))

            # Get action probabilities from actor
            action_probs = self.actor(imag_h.squeeze(0))
            action_dist = torch.distributions.Categorical(probs=action_probs)
            imag_action = action_dist.sample()
            imag_action_log_probs.append(action_dist.log_prob(imag_action))

            # Update imag_h and imag_s using the world model
            act_onehot = F.one_hot(imag_action, num_classes=self.act_dim).float()
            rnn_input = torch.cat([imag_s.detach(), act_onehot], dim=-1)
            rnn_input = rnn_input.unsqueeze(1)  # Shape: [batch_size, 1, input_size]

            # RNN step
            _, imag_h = self.world_model.rnn(rnn_input, imag_h)

            # Compute prior logits and sample imag_s
            prior_logits = self.world_model.prior_net(imag_h.squeeze(0))
            prior_logits = prior_logits.view(
                self.config.batch_size, self.config.latent_dim, self.config.latent_categories
            )
            imag_s = gumbel_softmax(prior_logits, tau=self.temperature, hard=False).view(self.config.batch_size, -1)

            # Predict reward
            decoder_input = torch.cat([imag_h.squeeze(0), imag_s], dim=-1)
            pred_reward = self.world_model.reward_decoder(decoder_input)
            pred_reward = symexp(pred_reward.squeeze(-1))
            imag_rewards.append(pred_reward)

        # Convert lists to tensors
        imag_states = torch.stack(imag_states)
        imag_rewards = torch.stack(imag_rewards)
        imag_action_log_probs = torch.stack(imag_action_log_probs)

        # Reshape imag_states for critic input
        imag_states_flat = imag_states.view(-1, self.config.hidden_dim)

        # Get value estimates for critic update (detach to prevent backprop through the world model)
        value_pred = self.critic(imag_states_flat.detach()).view(self.config.imagination_horizon, self.config.batch_size)

        # Calculate lambda-returns
        lambda_returns = torch.zeros_like(imag_rewards)
        last_value = value_pred[-1]
        for t in reversed(range(self.config.imagination_horizon)):
            if t == self.config.imagination_horizon - 1:
                bootstrap = last_value
            else:
                bootstrap = (1 - self.config.lambda_) * value_pred[t + 1] + self.config.lambda_ * lambda_returns[t + 1]
            lambda_returns[t] = imag_rewards[t] + self.config.gamma * bootstrap

        # Critic update
        value_target = lambda_returns.detach()
        critic_loss = F.mse_loss(value_pred, value_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        # Actor update
        # Recompute value estimates without detaching imag_states to allow gradients to flow
        value_pred_actor = self.critic(imag_states_flat).view(self.config.imagination_horizon, self.config.batch_size)
        advantage = (lambda_returns.detach() - value_pred_actor)
        actor_loss = -(advantage * imag_action_log_probs).mean()

        # Entropy regularization
        action_probs = self.actor(imag_states_flat)
        action_probs = action_probs.view(self.config.imagination_horizon, self.config.batch_size, -1)
        action_log_probs = torch.log(action_probs + 1e-8)
        entropy = -torch.sum(action_probs * action_log_probs, dim=-1).mean()
        actor_loss += -self.config.entropy_scale * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        # Update target critic
        self._soft_update(self.target_critic, self.critic)

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, target, source, tau=None):
        if tau is None:
            tau = self.config.tau
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def act(self, obs, reset=False):
        if reset:
            self.reset_hidden_states()

        if self.is_image:
            obs = torch.tensor(obs).float().to(device).unsqueeze(0)
        else:
            obs = self.obs_normalizer.normalize(obs)
            obs = torch.tensor(obs).float().to(device).unsqueeze(0)

        with torch.no_grad():
            # Encode observation
            obs_encoded = self.world_model.obs_encoder(obs)

            # Compute posterior over latent variables
            h = self.h.squeeze(0)  # Shape: [1, hidden_dim]
            posterior_input = torch.cat([h, obs_encoded], dim=-1)
            posterior_logits = self.world_model.posterior_net(posterior_input)
            posterior_logits = posterior_logits.view(
                1, self.config.latent_dim, self.config.latent_categories
            )
            posterior_sample = gumbel_softmax(posterior_logits, tau=self.temperature, hard=False).view(1, -1)

            # Get action probabilities from actor
            action_probs = self.actor(h)
            action_dist = torch.distributions.Categorical(probs=action_probs)
            action = action_dist.sample().cpu().numpy()[0]

            # Update hidden state
            act_onehot = F.one_hot(
                torch.tensor([action], device=device), num_classes=self.act_dim
            ).float()
            rnn_input = torch.cat([posterior_sample.detach(), act_onehot], dim=-1)
            _, self.h = self.world_model.rnn(rnn_input.unsqueeze(1), self.h)

        return action

    def store_transition(self, obs, act, rew, next_obs, done):
        if not self.is_image:
            self.obs_normalizer.update(obs)
            obs = self.obs_normalizer.normalize(obs)
            next_obs = self.obs_normalizer.normalize(next_obs)

        self.replay_buffer.store(obs, act, rew, next_obs, done)

    def train(self, num_updates):
        world_losses = []
        actor_losses = []
        critic_losses = []

        for _ in range(num_updates):
            world_loss = self.update_world_model()
            if world_loss is None:
                continue  # Not enough data to train
            losses = self.update_actor_and_critic()
            if losses is None:
                continue  # Not enough data to train
            actor_loss, critic_loss = losses

            world_losses.append(world_loss)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            # Anneal temperature
            self.temperature = max(self.temperature * self.config.temperature_decay, self.config.min_temperature)

        if len(world_losses) == 0:
            return None  # Not enough data to train

        return {
            'world_loss': np.mean(world_losses),
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses)
        }