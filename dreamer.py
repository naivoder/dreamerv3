import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import gym

from utils import gumbel_softmax, symlog, symexp, ObsNormalizer, ReplayBuffer
from networks import WorldModel, Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DreamerV3:
    def __init__(self, obs_shape, act_space, is_image, is_discrete, config):
        self.config = config
        self.obs_shape = obs_shape
        self.is_image = is_image
        self.is_discrete = is_discrete
        print("Is Discrete: ", self.is_discrete)
        
        if self.is_discrete:
            self.act_dim = act_space.n
        else:
            self.act_dim = act_space.shape[0]

        self.world_model = WorldModel(obs_shape, self.act_dim, is_image, is_discrete, config).to(device)
        self.actor = Actor(config.hidden_dim, self.act_dim, self.is_discrete, config).to(device)
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

        self.replay_buffer = ReplayBuffer(config.replay_buffer_capacity, obs_shape, self.act_dim)

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
        self.h = torch.zeros(1, 1, self.config.hidden_dim, device=device)

    def update_world_model(self):
        try:
            batch = self.replay_buffer.sample_batch(
                self.config.batch_size, self.config.seq_len
            )
        except ValueError:
            return  # Not enough data to train

        obs_seq, act_seq, rew_seq, next_obs_seq, done_seq = batch

        outputs = self.world_model(obs_seq, act_seq, tau=self.temperature)
        recon_obs = outputs["recon_obs"]
        pred_reward = outputs["pred_reward"]
        pred_discount = outputs["pred_discount"]
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

        # Discount prediction loss
        discount_target = (1.0 - done_seq.float()) * self.config.lambda_
        discount_loss = F.binary_cross_entropy(pred_discount, discount_target, reduction='none')
        discount_loss = discount_loss.mean()

        # Total loss
        loss_world = recon_loss + reward_loss + discount_loss + self.config.kl_scale * kl_loss

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

        # Encode observation
        obs_encoded = self.world_model.obs_encoder(obs)

        # Initialize hidden state
        imag_h = torch.zeros(1, self.config.batch_size, self.config.hidden_dim, device=device)

        # Compute initial posterior state
        posterior_input = torch.cat([imag_h.squeeze(0), obs_encoded], dim=-1)
        posterior_logits = self.world_model.posterior_net(posterior_input)
        posterior_logits = posterior_logits.view(
            self.config.batch_size, self.config.latent_dim, self.config.latent_categories
        )
        imag_s = gumbel_softmax(posterior_logits, tau=self.temperature, hard=False).view(self.config.batch_size, -1)

        imag_states = []
        imag_rewards = []
        imag_discounts = []
        imag_action_log_probs = []

        for _ in range(self.config.imagination_horizon):
            imag_states.append(imag_h.squeeze(0))

            # Get action probabilities from actor
            action_probs = self.actor(imag_h.squeeze(0))
            if self.is_discrete:
                action_dist = torch.distributions.Categorical(probs=action_probs)
                imag_action = action_dist.sample()
                imag_action_log_probs.append(action_dist.log_prob(imag_action))
            else:
                action_mean, action_std = action_probs
                action_dist = torch.distributions.Normal(action_mean, action_std)
                imag_action = action_dist.sample()
                imag_action_log_probs.append(action_dist.log_prob(imag_action))

            # Update imag_h and imag_s using the world model
            if self.is_discrete:
                act_onehot = F.one_hot(imag_action, num_classes=self.act_dim).float()
            else:
                act_onehot = imag_action.float()
            rnn_input = torch.cat([imag_s, act_onehot], dim=-1)
            rnn_input = rnn_input.unsqueeze(1)  # Shape: [batch_size, 1, input_size]

            # RNN step
            _, imag_h = self.world_model.rnn(rnn_input, imag_h)

            # Compute prior logits and sample imag_s
            prior_logits = self.world_model.prior_net(imag_h.squeeze(0))
            prior_logits = prior_logits.view(
                self.config.batch_size, self.config.latent_dim, self.config.latent_categories
            )
            imag_s = gumbel_softmax(prior_logits, tau=self.temperature, hard=False).view(self.config.batch_size, -1)

            # Predict reward and discount
            decoder_input = torch.cat([imag_h.squeeze(0), imag_s], dim=-1)
            pred_reward = self.world_model.reward_decoder(decoder_input)
            pred_reward = pred_reward.squeeze(-1)
            imag_rewards.append(pred_reward)

            pred_discount = self.world_model.discount_decoder(decoder_input)
            pred_discount = pred_discount.squeeze(-1)
            imag_discounts.append(pred_discount * self.config.lambda_)

        # Convert lists to tensors
        imag_states = torch.stack(imag_states)  # Shape: [imagination_horizon, batch_size, hidden_dim]
        imag_rewards = torch.stack(imag_rewards)  # Shape: [imagination_horizon, batch_size]
        imag_discounts = torch.stack(imag_discounts)  # Shape: [imagination_horizon, batch_size]
        imag_action_log_probs = torch.stack(imag_action_log_probs)  # Shape: [imagination_horizon, batch_size]

        # Reshape imag_states for critic input
        imag_states_flat = imag_states.view(-1, self.config.hidden_dim)

        # Get value estimates for critic update (detach to prevent backprop through the world model)
        value_pred = self.critic(imag_states_flat.detach()).view(self.config.imagination_horizon, self.config.batch_size)
        self.q_values = value_pred[0, :].detach().cpu().numpy()  # Save for logging

        # Bootstrap value is the last value prediction
        bootstrap = value_pred[-1]

        # Compute lambda returns
        lambda_returns = self.lambda_return(
            reward=imag_rewards,
            value=value_pred,
            pcont=imag_discounts,
            bootstrap=bootstrap,
            lambda_=self.config.lambda_,
        )

        self.returns = lambda_returns[0, :].detach().cpu().numpy()  # Save for logging

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
        if self.is_discrete:
            action_probs = action_probs.view(self.config.imagination_horizon, self.config.batch_size, -1)
            action_log_probs = torch.log(action_probs + 1e-8)
            entropy = -torch.sum(action_probs * action_log_probs, dim=-1).mean()
        else:
            action_mean, action_std = action_probs
            action_dist = torch.distributions.Normal(action_mean, action_std)
            entropy = action_dist.entropy().mean()
        actor_loss += -self.config.entropy_scale * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        # Update target critic using soft update
        self._soft_update(self.target_critic, self.critic)

        return actor_loss.item(), critic_loss.item()

    def lambda_return(self, reward, value, pcont, bootstrap, lambda_):
        next_values = torch.cat([value[1:], bootstrap[None]], 0)
        inputs = reward + pcont * next_values * (1 - lambda_)
        returns = self._static_scan(
            lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
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
            if self.is_discrete:
                action_dist = torch.distributions.Categorical(probs=action_probs)
                action = action_dist.sample().cpu().numpy()[0]
            else:
                action_mean, action_std = action_probs
                action_dist = torch.distributions.Normal(action_mean, action_std)
                action = action_dist.sample().cpu().numpy()

            # Update hidden state
            if self.is_discrete:
                act_onehot = F.one_hot(
                    torch.tensor([action], device=device), num_classes=self.act_dim
                ).float()
            else:
                act_onehot = torch.tensor(action, device=device).float().unsqueeze(0)
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