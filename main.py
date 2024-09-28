import gymnasium as gym
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio

from dreamer import DreamerV3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dreamer(config):
    env = gym.make(config.env)
    # Determine if the observation space is image-based or vector-based
    obs_space = env.observation_space
    is_image = isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3

    if is_image:
        # Preprocess observations to shape (3, 64, 64)
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )
        obs_shape = (3, 64, 64)
    else:
        obs_shape = obs_space.shape
        transform = None

    act_dim = env.action_space.n
    agent = DreamerV3(obs_shape, act_dim, is_image, config)
    total_rewards = []
    world_losses = []
    actor_losses = []
    critic_losses = []

    frame_idx = 0  # For temperature annealing
    avg_reward_window = 100  # Running average over the last 100 episodes
    best_avg_reward = float('-inf')
    best_weights = None

    with tqdm(total=config.episodes, desc="Training Progress", unit="episode") as pbar:
        for episode in range(config.episodes):
            obs, _ = env.reset()
            if is_image:
                obs = transform(obs).numpy()
            else:
                obs = obs.astype(np.float32)
            done = False
            episode_reward = 0
            agent.act(obs, reset=True)  # Reset hidden states at episode start

            while not done:
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if is_image:
                    next_obs_processed = transform(next_obs).numpy()
                    agent.store_transition(obs, action, reward, next_obs_processed, done)
                    obs = next_obs_processed
                else:
                    next_obs = next_obs.astype(np.float32)
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
                    # Update the progress bar with running average reward
                    if len(total_rewards) >= avg_reward_window:
                        running_avg_reward = sum(total_rewards[-avg_reward_window:]) / avg_reward_window
                    else:
                        running_avg_reward = sum(total_rewards) / len(total_rewards)

                    pbar.set_postfix({"Running Avg. Reward": f"{running_avg_reward:.2f}"})
                    pbar.update(1)  # Update the progress bar for one episode completion
                    break

            if len(total_rewards) >= avg_reward_window:
                running_avg_reward = sum(total_rewards[-avg_reward_window:]) / avg_reward_window
                if running_avg_reward > best_avg_reward:
                    best_avg_reward = running_avg_reward
                    best_weights = {
                        'world_model': agent.world_model.state_dict(),
                        'actor': agent.actor.state_dict(),
                        'critic': agent.critic.state_dict()
                    }

    # Plot losses and rewards
    plot_results(total_rewards, world_losses, actor_losses, critic_losses)

    # Create animation
    create_animation(env, agent, best_weights, config)

    return total_rewards


def plot_results(rewards, world_losses, actor_losses, critic_losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot rewards
    ax1.plot(rewards, label='Episode Reward')
    ax1.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), label='Running Average (100 episodes)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Rewards over Episodes')
    ax1.legend()

    # Plot losses
    ax2.plot(world_losses, label='World Model Loss')
    ax2.plot(actor_losses, label='Actor Loss')
    ax2.plot(critic_losses, label='Critic Loss')
    ax2.set_xlabel('Update Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Losses over Update Steps')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('dreamer_results.png')
    plt.close()


def create_animation(env, agent, best_weights, filename="dreamer_animation.gif"):
    # Load best weights
    agent.world_model.load_state_dict(best_weights['world_model'])
    agent.actor.load_state_dict(best_weights['actor'])
    agent.critic.load_state_dict(best_weights['critic'])

    obs, _ = env.reset()
    frames = []

    for _ in range(1000):  # Adjust the number of steps as needed
        frames.append(env.render())
        action = agent.act(obs)
        obs, _, done, _, _ = env.step(action)
        if done:
            break

    env.close()

    # Save animation as GIF
    with imageio.get_writer(filename, mode="I", loop=0) as writer:
        for frame in frames:
            writer.append_data(frame)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--train_horizon", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--latent_categories", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--actor_lr", type=float, default=3e-5)
    parser.add_argument("--critic_lr", type=float, default=3e-5)
    parser.add_argument("--world_lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--imagination_horizon", type=int, default=15)
    parser.add_argument("--free_nats", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--replay_buffer_capacity", type=int, default=100000)
    parser.add_argument("--entropy_scale", type=float, default=1e-3)
    parser.add_argument("--kl_balance_alpha", type=float, default=0.8)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=100.0)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--num_updates", type=int, default=5)
    parser.add_argument("--min_buffer_size", type=int, default=5000)
    parser.add_argument("--init_temperature", type=float, default=1.0)
    parser.add_argument("--temperature_decay", type=float, default=0.999)
    parser.add_argument("--min_temperature", type=float, default=0.2)
    parser.add_argument("--actor_temperature", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--kl_scale", type=float, default=1.0)
    args = parser.parse_args()

    train_dreamer(args)
