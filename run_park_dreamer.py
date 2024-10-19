import gymnasium as gym
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import park
from dreamer import DreamerV3
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dreamer(config):
    env = park.make(config.env)

    obs_space = env.observation_space
    act_space = env.action_space
    
    is_discrete = isinstance(act_space, gym.spaces.Discrete) or isinstance(act_space, park.spaces.Discrete)

    transform = None
    # Get the observation shape from the environment
    obs = env.reset()
    obs_shape = obs.shape if hasattr(obs, 'shape') else (len(obs),)

    # Print observation and action space information for debugging
    print(f"Observation space: {obs_space}")
    print(f"Observation shape: {obs_shape}")
    print(f"Action space: {act_space}")

    agent = DreamerV3(obs_shape, env.action_space, False, is_discrete, config)

    total_rewards = []
    world_losses = []
    actor_losses = []
    critic_losses = []

    frame_idx = 0
    avg_reward_window = 100
    best_avg_reward = float('-inf')
    best_weights = None

    with tqdm(total=config.episodes, desc=f"Training {config.env}", unit="episode") as pbar:
        for _ in range(config.episodes):
            obs = env.reset()
            obs = np.array(obs).astype(np.float32)

            done = False
            episode_reward = 0
            agent.act(obs, reset=True)

            while not done:
                action = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                next_obs = np.array(next_obs).astype(np.float32)
                agent.store_transition(obs, action, reward, next_obs, done)
                obs = next_obs

                episode_reward += reward
                frame_idx += 1

                if len(agent.replay_buffer) > config.min_buffer_size:
                    if frame_idx % config.train_horizon == 0:
                        losses = agent.train(num_updates=config.num_updates)
                        if losses is not None:
                            world_losses.append(losses['world_loss'])
                            actor_losses.append(losses['actor_loss'])
                            critic_losses.append(losses['critic_loss'])

                if done:
                    total_rewards.append(episode_reward)
                    if len(total_rewards) >= avg_reward_window:
                        running_avg_reward = sum(total_rewards[-avg_reward_window:]) / avg_reward_window
                    else:
                        running_avg_reward = sum(total_rewards) / len(total_rewards)

                    if running_avg_reward > best_avg_reward:
                        best_avg_reward = running_avg_reward
                        best_weights = {
                            'world_model': agent.world_model.state_dict(),
                            'actor': agent.actor.state_dict(),
                            'critic': agent.critic.state_dict()
                        }

                    pbar.set_postfix({"Running Avg. Reward": f"{running_avg_reward:.2f}"})
                    pbar.update(1)
                    break

    plot_results(total_rewards, world_losses, actor_losses, critic_losses, config.env)
    save_best_weights(best_weights, config.env)

    return total_rewards


def plot_results(rewards, world_losses, actor_losses, critic_losses, env_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(rewards, label='Episode Reward')
    ax1.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), label='Running Average (100 episodes)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title(f'Rewards over Episodes for {env_name}')
    ax1.legend()

    ax2.plot(world_losses, label='World Model Loss')
    ax2.plot(actor_losses, label='Actor Loss')
    ax2.plot(critic_losses, label='Critic Loss')
    ax2.set_xlabel('Update Step')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Losses over Update Steps for {env_name}')
    ax2.legend()

    plt.tight_layout()
    plot_filename = f'{env_name}.png'
    plt.savefig(plot_filename)
    plt.close()


def save_best_weights(best_weights, env_name):
    if best_weights:
        os.makedirs('weights', exist_ok=True)
        torch.save(best_weights, f'weights/{env_name}.pt')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--train_horizon", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--latent_categories", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--actor_lr", type=float, default=3e-5)
    parser.add_argument("--critic_lr", type=float, default=3e-5)
    parser.add_argument("--world_lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--imagination_horizon", type=int, default=15)
    parser.add_argument("--free_nats", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--replay_buffer_capacity", type=int, default=100000)
    parser.add_argument("--entropy_scale", type=float, default=1e-3)
    parser.add_argument("--kl_balance_alpha", type=float, default=0.8)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=100.0)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--num_updates", type=int, default=3)
    parser.add_argument("--min_buffer_size", type=int, default=5000)
    parser.add_argument("--init_temperature", type=float, default=1.0)
    parser.add_argument("--temperature_decay", type=float, default=0.999)
    parser.add_argument("--min_temperature", type=float, default=0.5)
    parser.add_argument("--actor_temperature", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--kl_scale", type=float, default=1.0)
    args = parser.parse_args()

    print(f"Training on environment: {args.env}")
    train_dreamer(args)
