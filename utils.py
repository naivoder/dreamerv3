import numpy as np
import torch
from torch import nn
import imageio
import wandb
import os


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


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def save_animation(frames, filename):
    with imageio.get_writer(filename, mode="I", loop=0) as writer:
        for frame in frames:
            writer.append_data(frame)


def create_animation(env, agent, save_prefix, mod="best", seeds=10):
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

    save_animation(best_frames, f"environments/{save_prefix}.gif")
    env.close()


def log_hparams(config, run_name):
    with open(config.wandb_key, "r", encoding="utf-8") as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
    config.key_path = None

    wandb.init(
        project="dreamerv3-atari", name=run_name, config=vars(config), reinit=True
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
            # "Entropy/Reward": losses["reward_entropy"],
        },
        step=ep,
    )


def log_rewards(
    ep: int,
    score: float,
    avg_score: float,
    buffer_len: int,
    total_episodes: int,
):
    wandb.log(
        {
            "Reward/Score": score,
            "Reward/Average": avg_score,
            "Buffer/Length": buffer_len,
        },
        step=ep,
    )

    e_str = f"[Ep {ep:05d}/{total_episodes}]"
    s_str = f"Score = {score:8.2f}"
    a_str = f"Avg.Score = {avg_score:8.2f}"
    b_str = f"Mem.Length = {buffer_len:07d}"
    print(f"{e_str}  {s_str}   {a_str}   {b_str}", end="\r")
