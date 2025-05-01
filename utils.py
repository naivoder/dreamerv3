import numpy as np
import torch
from torch import nn
import imageio


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


def save_animation(frames, filename):
    with imageio.get_writer(filename, mode="I", loop=0) as writer:
        for frame in frames:
            writer.append_data(frame)


def create_animation(env, agent, save_prefix, mod="best", seeds=10):
    agent.load_checkpoint(f"weights/{save_prefix}_{mod}_dreamerv3.pt")
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


def log_losses(writer, ep, losses):
    writer.add_scalar("Loss/World", losses["world_loss"], ep)
    writer.add_scalar("Loss/Recon", losses["recon_loss"], ep)
    writer.add_scalar("Loss/Reward", losses["reward_loss"], ep)
    writer.add_scalar("Loss/Continue", losses["continue_loss"], ep)
    writer.add_scalar("Loss/KL", losses["kl_loss"], ep)
    writer.add_scalar("Loss/Actor", losses["actor_loss"], ep)
    writer.add_scalar("Loss/Critic", losses["critic_loss"], ep)
    writer.add_scalar("Entropy/Actor", losses["actor_entropy"], ep)
    writer.add_scalar("Entropy/Prior", losses["prior_entropy"], ep)
    writer.add_scalar("Entropy/Posterior", losses["posterior_entropy"], ep)
    writer.add_scalar("Entropy/Reward", losses["reward_entropy"], ep)


def log_rewards(writer, ep, score, avg_score, buffer_len, total_episodes):
    writer.add_scalar("Reward/Score", score, ep)
    writer.add_scalar("Reward/Average", avg_score, ep)
    writer.add_scalar("Buffer/Length", buffer_len, ep)

    e_str = f"[Ep {ep:05d}/{total_episodes}]"
    s_str = f"Score = {score:8.2f}"
    a_str = f"Avg.Score = {avg_score:8.2f}"
    b_str = f"Mem.Length = {buffer_len:07d}"
    print(f"{e_str} {s_str}  {a_str}  {b_str}", end="\r")
