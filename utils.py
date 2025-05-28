import numpy as np
import torch
from torch import nn
import imageio
import wandb
import os
import gymnasium as gym
import time


def make_env(
    env_name, record_video=False, video_folder="videos", video_interval=100, test=False
):
    env = gym.make(env_name, render_mode="rgb_array" if record_video else None)

    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=1,
        screen_size=64,
        grayscale_obs=False,
        scale_obs=True,
        noop_max=0 if test else 30,
    )
    env = gym.wrappers.TransformObservation(
        env, lambda obs: np.transpose(obs, (2, 0, 1)), None
    )
    env.observation_space = gym.spaces.Box(
        low=0, high=1, shape=(3, 64, 64), dtype=np.float32
    )

    if record_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda x: x % video_interval == 0,
            name_prefix=env_name.split("/")[-1],
        )

    return env


def make_vec_env(env_name, num_envs=16, video_folder="videos"):
    os.makedirs(video_folder, exist_ok=True)

    env_fns = [
        lambda i=i: make_env(
            env_name,
            record_video=(i == 0),
            video_folder=video_folder,
            video_interval=1000,
        )
        for i in range(num_envs)
    ]

    vec_env = gym.vector.AsyncVectorEnv(env_fns)
    return vec_env


class VideoLoggerWrapper(gym.vector.VectorWrapper):
    def __init__(self, env, video_folder, get_step_callback):
        super().__init__(env)
        self.video_folder = video_folder
        self.last_logged = 0
        self.get_step = get_step_callback

    def step(self, action):
        obs, rewards, terminated, truncated, infos = super().step(action)

        current_step = self.get_step()

        new_videos = [
            f
            for f in os.listdir(self.video_folder)
            if f.endswith(".mp4")
            and os.path.getmtime(os.path.join(self.video_folder, f)) > self.last_logged
        ]

        for video_file in sorted(
            new_videos,
            key=lambda x: os.path.getctime(os.path.join(self.video_folder, x)),
        ):
            video_path = os.path.join(self.video_folder, video_file)
            wandb.log({"video": wandb.Video(video_path)}, step=current_step)
            os.remove(video_path)
            self.last_logged = time.time()

        return obs, rewards, terminated, truncated, infos


def preprocess(image):
    return image.astype(np.float32) / 255.0


def quantize(image):
    return (image * 255).clip(0, 255).astype(np.uint8)


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1e-5)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.clamp(torch.abs(x), max=20.0)) - 1)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def adaptive_gradient_clip(model, clip_factor=0.3, eps=1e-3):
    for param in model.parameters():
        if param.grad is not None:
            weight_norm = torch.norm(param.detach(), p=2)  # L2 norm of weights
            grad_norm = torch.norm(param.grad.detach(), p=2)  # L2 norm of gradients
            max_norm = clip_factor * weight_norm + eps
            if grad_norm > max_norm:
                scale = max_norm / (grad_norm + 1e-8)  # Avoid division by zero
                param.grad.mul_(scale)  # Scale gradients in-place


def save_animation(frames, filename):
    with imageio.get_writer(filename, mode="I", loop=0) as writer:
        for frame in frames:
            writer.append_data(frame)


def create_animation(env, agent, save_prefix, seeds=10):
    for mod in ["best", "best_avg", "final"]:
        save_path = f"environments/{save_prefix}_{mod}.gif"
        agent.load_checkpoint(save_prefix, mod)
        best_total_reward, best_frames = float("-inf"), None

        for _ in range(seeds):
            state, _ = env.reset()
            frames, total_reward = [], 0
            term, trunc = False, False

            while not (term or trunc):
                frames.append(env.render())
                action = agent.act(state)
                next_state, reward, term, trunc, _ = env.step(action)
                total_reward += reward
                state = next_state

            if total_reward > best_total_reward:
                best_total_reward = total_reward
                best_frames = frames

        save_animation(best_frames, save_path)
        wandb.log({f"Animation/{mod}": wandb.Video(save_path, format="gif")})


def log_hparams(config, run_name):
    with open(config.wandb_key, "r", encoding="utf-8") as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()

    wandb.init(
        project="dreamerv3-atari-v2",
        name=run_name,
        config=wandb.helper.parse_config(config, exclude=("wandb_key",)),
        save_code=True,
    )


def log_losses(ep: int, losses: dict):
    wandb.log(
        {
            "Loss/World": losses["world_loss"],
            "Loss/Recon": losses["recon_loss"],
            "Loss/Reward": losses["reward_loss"],
            "Loss/Continue": losses["continue_loss"],
            "Loss/KL": losses["kl_loss"],
            "Loss/Actor": losses["actor_loss"],
            "Loss/Critic": losses["critic_loss"],
            "Entropy/Actor": losses["actor_entropy"],
            "Entropy/Prior": losses["prior_entropy"],
            "Entropy/Posterior": losses["posterior_entropy"],
        },
        step=ep,
    )


def log_rewards(
    step: int,
    avg_score: float,
    best_score: float,
    mem_size: int,
    episode: int,
    total_episodes: int,
):
    wandb.log(
        {
            "Reward/Average": avg_score,
            "Reward/Best": best_score,
            "Memory/Size": mem_size,
        },
        step=step,
    )

    e_str = f"[Ep {episode:05d}/{total_episodes}]"
    a_str = f"Avg.Score = {avg_score:8.2f}"
    b_str = f"Best.Score = {best_score:8.2f}"
    s_str = f"Step = {step:8d}"
    m_str = f"Mem.Size = {mem_size:7d}"
    print(f"{e_str} {a_str} {b_str} {m_str} {s_str}", end="\r")
