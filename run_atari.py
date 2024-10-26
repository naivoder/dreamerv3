import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from ale_py import ALEInterface, LoggerMode
from config import environments
from preprocess import AtariEnv
from dreamer import DreamerV3
import utils

warnings.simplefilter("ignore")
ALEInterface.setLoggerMode(LoggerMode.Error)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_dreamer(args):
    def make_env():
        return AtariEnv(
            args.env,
            shape=(42, 42),
            repeat=4,
            clip_rewards=True,
        ).make()

    save_prefix = args.env.split("/")[-1]
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(args.n_envs)])
    obs_shape = envs.single_observation_space.shape
    act_dim = envs.single_action_space.n
    is_image = len(obs_shape) == 3

    print(f"\nEnvironment: {save_prefix}")
    print(f"Obs.Space: {envs.single_observation_space.shape}")
    print(f"Act.Space: {envs.single_action_space.n}")

    agent = DreamerV3(obs_shape, act_dim, is_image, True, args)
    world_losses, actor_losses, critic_losses = [], [], []
    best_avg_reward, frame_idx, avg_reward_window = float("-inf"), 0, 100
    score = np.zeros(args.n_envs)
    history = []
    states, _ = envs.reset()
    done = np.zeros(args.n_envs, dtype=bool)

    while len(history) < args.episodes:
        actions = [
            agent.act(state, env_id=j, reset=done[j]) for j, state in enumerate(states)
        ]

        next_states, rewards, term, trunc, _ = envs.step(actions)
        done = np.logical_or(term, trunc)

        for j in range(args.n_envs):
            agent.store_transition(
                j, states[j], actions[j], rewards[j], next_states[j], done[j]
            )
            score[j] += rewards[j]
            if done[j]:
                history.append(score[j])
                score[j] = 0
                agent.reset_hidden_states(env_id=j)

        states = next_states

        if (
            len(agent.replay_buffer) > args.min_buffer_size
            and frame_idx % args.train_horizon == 0
        ):
            losses = agent.train(num_updates=args.num_updates)
            if losses:
                world_losses.append(losses["world_loss"])
                actor_losses.append(losses["actor_loss"])
                critic_losses.append(losses["critic_loss"])

        frame_idx += args.n_envs
        if len(history) > 0:
            avg_score = np.mean(history[-avg_reward_window:])

            if avg_score > best_avg_reward:
                best_avg_reward = avg_score
                agent.save_checkpoint()

            ep_str = f"[Episode {len(history):05}/{args.episodes}]"
            avg_str = f"  Avg.Score = {avg_score:.2f}"
            print(ep_str + avg_str, end="\r")

    torch.save(
        agent.world_model.state_dict(), f"weights/{save_prefix}_world_model_final.pth"
    )
    torch.save(agent.actor.state_dict(), f"weights/{save_prefix}_actor_final.pth")
    torch.save(agent.critic.state_dict(), f"weights/{save_prefix}_critic_final.pth")

    plot_results(history, world_losses, actor_losses, critic_losses, save_prefix)
    create_animation(args.env, agent)


def plot_results(rewards, world_losses, actor_losses, critic_losses, save_prefix):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(rewards, label="Episode Reward")
    ax1.plot(
        np.convolve(rewards, np.ones(100) / 100, mode="valid"),
        label="Running Average (100 episodes)",
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Rewards over Episodes")
    ax1.legend()

    ax2.plot(world_losses, label="World Model Loss")
    ax2.plot(actor_losses, label="Actor Loss")
    ax2.plot(critic_losses, label="Critic Loss")
    ax2.set_xlabel("Update Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("Losses over Update Steps")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"results/{save_prefix}.png")
    plt.close()


def create_animation(env_name, agent, seeds=100):
    agent.load_checkpoint()
    env = AtariEnv(
        env_name,
        shape=(42, 42),
        repeat=4,
        clip_rewards=False,
    ).make()

    save_prefix = env_name.split("/")[-1]
    best_total_reward, best_frames = float("-inf"), None

    for s in range(seeds):
        state, _ = env.reset()
        frames, total_reward = [], 0

        term, trunc = False, False
        while not term and not trunc:
            frames.append(env.render())
            action = agent.choose_action(state)
            next_state, reward, term, trunc, _ = env.step(action)

            total_reward += reward
            state = next_state

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_frames = frames

    utils.save_animation(best_frames, f"environments/{save_prefix}.gif")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--n_envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--train_horizon", type=int, default=1000)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--latent_categories", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--world_lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--imagination_horizon", type=int, default=15)
    parser.add_argument("--free_nats", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--replay_buffer_capacity", type=int, default=100000)
    parser.add_argument("--entropy_scale", type=float, default=1e-3)
    parser.add_argument("--kl_balance_alpha", type=float, default=0.8)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=100.0)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--num_updates", type=int, default=10)
    parser.add_argument("--min_buffer_size", type=int, default=10000)
    parser.add_argument("--init_temperature", type=float, default=1.0)
    parser.add_argument("--temperature_decay", type=float, default=0.999)
    parser.add_argument("--min_temperature", type=float, default=0.5)
    parser.add_argument("--actor_temperature", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--kl_scale", type=float, default=1.0)
    args = parser.parse_args()

    for fname in ["metrics", "environments", "weights", "csv"]:
        if not os.path.exists(fname):
            os.makedirs(fname)

    if args.env:
        train_dreamer(args)
    else:
        for env_name in environments:
            args.env = env_name
            train_dreamer(args)
